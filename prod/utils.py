import os
import urllib.request
from zipfile import ZipFile
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

# URLs del dataset y pesos
DATASET_ZIP_URL = "https://www.dropbox.com/scl/fi/buygp1u3dlvql1omlgii6/title30cat.zip?rlkey=emlr8c439whnexhezqjanlr19&st=58w2deja&dl=1"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "224x224")
CSV_TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "book30-listing-train.csv")
ZIP_PATH = os.path.join(BASE_DIR, "..", "data", "title30cat.zip")
WEIGHTS_PATH_CGAN = os.path.join(BASE_DIR, "modelo_cgan.pth")
WEIGHTS_PATH_CVAE = os.path.join(BASE_DIR, "modelo_cvae.pth")
EXTRACT_PATH = os.path.join(BASE_DIR, "..", "data")

class LibrosDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str):
        """
        Dataset para imágenes de portadas con etiquetas desde un CSV.

        Args:
            csv_path (str): Ruta al archivo CSV con columnas 'filename' y 'genre'.
            images_dir (str): Carpeta donde están las imágenes.
        """
        self.data = pd.read_csv(csv_path, delimiter=";")
        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), # 224??
            transforms.ToTensor(),
            # transforms.Normalize([0.5]*3, [0.5]*3)
            ])

        # Crear un mapeo de género a índice y viceversa
        self.genres = sorted(self.data['Category'].unique())
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['Filename'])
        label_idx = self.genre_to_idx[row['Category']]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label_tensor

@st.cache_resource
def load_dataloader(batch_size=64):
    dataset = LibrosDataset(csv_path=CSV_TRAIN_PATH, images_dir=DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def download_and_extract_data():
    """Downloads and extracts the dataset and CSV if they don't exist."""
    os.makedirs("../data", exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        print("Downloading and extracting dataset...")
        urllib.request.urlretrieve(DATASET_ZIP_URL, ZIP_PATH)
        with ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        os.remove(ZIP_PATH)
        print("Dataset downloaded and extracted.")


@st.cache_data
def load_dataframe():
    """Loads the training data into a pandas DataFrame."""
    try:
        df = pd.read_csv(CSV_TRAIN_PATH, delimiter=";")
        print("DataFrame loaded successfully.")
        return df
    except FileNotFoundError:
        print(
            f"Error: {CSV_TRAIN_PATH} not found. Run download_and_extract_data() first."
        )
        return None
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return None


def obtener_id_genero(dataframe, genre_name):
    """Devuelve el índice numérico del género a partir del nombre."""
    genres = sorted(dataframe["Category"].unique())
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    return genre_to_idx.get(genre_name, -1)

#Clase generador de la CGAN
class Generator(nn.Module):
    def __init__(self, z_dim, genre_dim, img_channels=3, feature_g=64):
        super().__init__()
        input_dim = z_dim + genre_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, feature_g * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.net(x)

#CVAE
class CVAE(nn.Module):
    def __init__(self, z_dim=64, num_classes=10):
        super(CVAE, self).__init__()
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + num_classes, 32, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),               # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),              # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),             # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, z_dim)

        # Decoder
        self.fc_decode = nn.Linear(z_dim + num_classes, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 32x32 -> 64x64
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        one_hot_expanded = one_hot.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, one_hot_expanded], dim=1)

        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        z_cond = torch.cat([z, one_hot], dim=1)
        h_dec = self.fc_decode(z_cond)
        x_recon = self.decoder(h_dec)
        return x_recon, mu, logvar

@st.cache_resource
def load_model_cgan():
    """Loads the generator with pretrained weights."""

    device = torch.device("cpu")
    z_dim = 100
    num_classes = 30
    G = Generator(z_dim, num_classes).to(device)

    checkpoint = torch.load(WEIGHTS_PATH_CGAN, map_location=device)
    G.load_state_dict(checkpoint["generator_state_dict"])
    G.eval()
    return G

@st.cache_resource
def load_model_cvae():
    """Carga el modelo CVAE con los pesos preentrenados."""

    device = torch.device("cpu")
    z_dim = 64
    num_classes = 30
    model = CVAE(z_dim=z_dim, num_classes=num_classes).to(device)

    checkpoint = torch.load(WEIGHTS_PATH_CVAE, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generar_por_genero_cgan(generator, genre_id, num_imgs=8):
    """Genera una lista de imágenes falsas (tensors) para un género dado."""
    z_dim = 100
    num_classes = 30
    device = torch.device("cpu")
    generator.eval()
    with torch.no_grad():
        labels = torch.full((num_imgs,), genre_id, dtype=torch.long, device=device)
        labels_one_hot = (
            torch.nn.functional.one_hot(labels, num_classes=num_classes)
            .float()
            .to(device)
        )
        z = torch.randn(num_imgs, z_dim, device=device)
        fake_imgs = generator(z, labels_one_hot).detach().cpu()
        return fake_imgs

def generar_por_genero_cvae(model, genre_id, num_imgs=8):
    """
    Genera imágenes falsas (tensors) para un género dado usando el CVAE.

    model: modelo CVAE ya cargado.
    genre_id: ID del género deseado.
    num_imgs: cantidad de imágenes a generar.

    Retorna:
        Tensor con las imágenes generadas (shape: [num_imgs, 3, 64, 64]).
    """
    z_dim = 64
    num_classes = 30
    device = torch.device("cpu")
    model.eval()

    # Carga el dataloader dentro de la función
    dataloader = load_dataloader(batch_size=128)

    with torch.no_grad():
        latent_vectors = []

        # Recolectar vectores latentes a partir de datos reales del género
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)
            genre_indices = (labels == genre_id).nonzero(as_tuple=True)[0]
            if genre_indices.numel() > 0:
                x_genre = x[genre_indices]
                labels_genre = labels[genre_indices]

                one_hot = F.one_hot(labels_genre, num_classes=num_classes).float()
                one_hot_expanded = one_hot.unsqueeze(2).unsqueeze(3).expand(-1, -1, x_genre.size(2), x_genre.size(3))
                x_cond = torch.cat([x_genre, one_hot_expanded], dim=1)

                h = model.encoder(x_cond)
                mu = model.fc_mu(h)
                latent_vectors.append(mu.cpu())

                if torch.cat(latent_vectors).size(0) >= num_imgs:
                    break

        if not latent_vectors:
            print(f"No se encontraron vectores latentes para el género {genre_id}.")
            return None

        all_latent_vectors = torch.cat(latent_vectors)[:num_imgs].to(device)
        labels_for_decoder = torch.full((num_imgs,), genre_id, dtype=torch.long).to(device)
        one_hot_for_decoder = F.one_hot(labels_for_decoder, num_classes=num_classes).float()

        z_cond = torch.cat([all_latent_vectors, one_hot_for_decoder], dim=1)
        h_dec = model.fc_decode(z_cond)
        fake_imgs = model.decoder(h_dec).detach().cpu()

        return fake_imgs

def normalize_individual_images(imgs):
    imgs_normalized = torch.empty_like(imgs)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            imgs_normalized[i] = (img - img_min) / (img_max - img_min)
        else:
            imgs_normalized[i] = torch.zeros_like(img)
    return imgs_normalized
