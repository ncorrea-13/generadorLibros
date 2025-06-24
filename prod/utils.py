import os
import urllib.request
from zipfile import ZipFile
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

# URLs del dataset y pesos
DATASET_ZIP_URL = "https://www.dropbox.com/scl/fi/buygp1u3dlvql1omlgii6/title30cat.zip?rlkey=emlr8c439whnexhezqjanlr19&st=58w2deja&dl=1"
CSV_TRAIN_URL = "https://www.dropbox.com/scl/fi/b28fi1cd6k4vtaj0en1lu/book30-listing-train.csv?rlkey=dd4lrdkoleiedezjzx9ing1yb&st=jwcx2hys&dl=1"
WEIGHTS_URL = "https://www.dropbox.com/scl/fi/9n666urcnw09xbz047e2d/cgan_checkpoint.pth?rlkey=3y6zhkq380ndayxpo0h2i0fxq&st=m492m9qb&dl=1"

# Paths
DATASET_PATH = "../data/224x224"
CSV_TRAIN_PATH = "../data/book30-listing-train.csv"
ZIP_PATH = "../data/title30cat.zip"
WEIGHTS_PATH_CGAN = "./modelo_cgan.pth"
WEIGHTS_PATH_CVAE = "./modelo_cvae.pth"


def download_and_extract_data():
    """Downloads and extracts the dataset and CSV if they don't exist."""
    os.makedirs("../data", exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        print("Downloading and extracting dataset...")
        urllib.request.urlretrieve(DATASET_ZIP_URL, ZIP_PATH)
        with ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall("../data")
        os.remove(ZIP_PATH)
        print("Dataset downloaded and extracted.")

    if not os.path.exists(CSV_TRAIN_PATH):
        print("Downloading CSV file...")
        urllib.request.urlretrieve(CSV_TRAIN_URL, CSV_TRAIN_PATH)
        print("CSV file downloaded.")


def download_weights():
    """Downloads the model weights if not present."""
    if not os.path.exists(WEIGHTS_PATH_CGAN):
        print("Downloading model weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH_CGAN)
        print("Weights downloaded.")
    if not os.path.exists(WEIGHTS_PATH_CVAE):
        print("Downloading model weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH_CVAE)
        print("Weights downloaded.")


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


@st.cache_resource
def load_model():
    """Loads the generator with pretrained weights."""
    download_weights()

    device = torch.device("cpu")
    z_dim = 100
    num_classes = 30
    G = Generator(z_dim, num_classes).to(device)

    checkpoint = torch.load(WEIGHTS_PATH_CGAN, map_location=device)
    G.load_state_dict(checkpoint["generator_state_dict"])
    G.eval()
    return G


def generar_por_genero(generator, genre_id, num_imgs=8):
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
