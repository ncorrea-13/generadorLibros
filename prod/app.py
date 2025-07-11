import streamlit as st
import pandas as pd
import os
import random
from utils import (
download_and_extract_data,
load_dataframe,
DATASET_PATH,
obtener_id_genero,
load_model_cgan,
generar_por_genero_cgan,
normalize_individual_images,
load_model_cvae,
generar_por_genero_cvae
)

# ---------- Estilos comunes ----------
st.markdown(
    """
    <style>
        .image-caption {
          margin-top: 0px;
          font-size: 16px;
          text-align: center;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 100%;
          display: block;
      }

        .stButton > button {
            display: block;
            margin: 0 auto;
            border: none;
            background-color: #1f77b4;
            color: white;
            padding: 6px 12px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #105a8b;
        }
        .stButton > button:focus {
            outline: none;
            box-shadow: none;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------- Carga inicial ----------
download_and_extract_data()
df = load_dataframe()
cGAN_generator = load_model_cgan()
cVAE = load_model_cvae()
GENRES = df["Category"].unique()

if "pantalla" not in st.session_state:
    st.session_state.pantalla = "inicio"

# Sidebar activado si no está en la pantalla de inicio
if st.session_state.pantalla != "inicio":
    st.sidebar.title("Navegación")
    if st.sidebar.button("🏠 Volver al inicio"):
        st.session_state.pantalla = "inicio"
        st.rerun()
    if st.sidebar.button("🎨 Ir al Generador"):
        st.session_state.pantalla = "generador"
        st.rerun()
    if st.sidebar.button("❓ Ir al Juego"):
        st.session_state.pantalla = "juego"
        st.rerun()

# ---------- Pantalla de Inicio ----------
if st.session_state.pantalla == "inicio":

    # ---------- Estilos del Inicio ----------
    st.markdown("""
        <style>
            .stApp {
                background-image: url("https://image.made-in-china.com/202f0j00MCpROtwEEJbh/Deep-Dark-Brown-High-Quality-with-IXPE-Wood-Texture-Spc-Flooring-3-5-6mm.webp");
                background-size: cover;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 BookCoverAI - Creatividad y Juego con IA")
    st.markdown("""
    Bienvenido a **BookCoverAI**, una aplicación interactiva que combina inteligencia artificial y libros.

    - 🎨 **Generador de Portadas de Libro**: Seleccioná un género y generá portadas con estilo.
    - ❓ **Adiviná el Género**: ¡Jugá a adivinar el género de las portadas generadas por IA!
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎨 Generador de Portadas de Libro"):
            st.session_state.pantalla = "generador"
            st.rerun()
    with col2:
        if st.button("❓ Adiviná el Género"):
            st.session_state.pantalla = "juego"
            st.rerun()

# ---------- Generador de Portadas ----------
elif st.session_state.pantalla == "generador":

    # ---------- Estilos del Generador ----------
    st.markdown("""
        <style>
            .stApp {
                background-image: url("https://www.transparenttextures.com/patterns/white-wall.png");
                background-size: cover;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎨 Generador de Portadas de Libro")
    IMAGES_PER_PAGE = 15
    IMAGES_PER_ROW = 5

    if "page_number" not in st.session_state:
        st.session_state.page_number = 0

    if 'samples' not in st.session_state:
        genres = df["Category"].unique()
        rows = [df[df["Category"] == genre].sample(1).iloc[0] for genre in genres]
        st.session_state.samples = pd.DataFrame(rows)

    samples = st.session_state.samples
    total_pages = (len(samples) - 1) // IMAGES_PER_PAGE + 1
    start_idx = st.session_state.page_number * IMAGES_PER_PAGE
    end_idx = start_idx + IMAGES_PER_PAGE
    current_samples = samples.iloc[start_idx:end_idx]

    cols = st.columns(IMAGES_PER_ROW)

    for i, (idx, row) in enumerate(current_samples.iterrows()):
        img_path = os.path.join(DATASET_PATH, row["Filename"])
        genre = row["Category"]
        with cols[i % IMAGES_PER_ROW]:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
                st.markdown(f"<div class='image-caption'>{genre}</div>", unsafe_allow_html=True)
                if st.button("Seleccionar", key=f"select_{start_idx + i}"):
                  if st.session_state.get("selected_genre") != genre:
                      st.session_state["selected_genre"] = genre
                      st.session_state.pop("generated_imgs", None)
            else:
                st.warning(f"Image file not found: {img_path}")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.page_number > 0:
            if st.button("⬅️ Anterior"):
                st.session_state.page_number -= 1
                st.rerun()
    with col3:
        if st.session_state.page_number < total_pages - 1:
            if st.button("Siguiente ➡️"):
                st.session_state.page_number += 1
                st.rerun()

    if "selected_genre" in st.session_state:
        st.success(f"Género seleccionado: {st.session_state.selected_genre}")
        genre_id = obtener_id_genero(df, st.session_state.selected_genre)

        if genre_id != -1:
            cant_images = st.number_input("Indique la cantidad de imágenes a generar:", min_value=1, max_value=12, step=1, value=1)
            modelo_seleccionado = st.selectbox("Seleccione el generador a utilizar:", ["CGAN", "CVAE"])

            if st.button("🎨 Generar Imágenes"):
                if modelo_seleccionado == "CGAN":
                    imgs = generar_por_genero_cgan(cGAN_generator, genre_id, cant_images)
                else:
                    imgs = generar_por_genero_cvae(cVAE, genre_id, cant_images)

                imgs = normalize_individual_images(imgs)
                st.session_state.generated_imgs = imgs

    if "generated_imgs" in st.session_state:
        st.markdown("### Imágenes generadas")
        cols = st.columns(4)
        for i, img in enumerate(st.session_state.generated_imgs):
            with cols[i % 4]:
                img = img.permute(1, 2, 0).numpy()
                st.image(img, use_container_width=True)
                st.markdown(f"<div class='image-caption'>Imagen {i+1}</div>", unsafe_allow_html=True)

# ---------- Juego: Adiviná el Género ----------
elif st.session_state.pantalla == "juego":

    # ---------- Estilos del Juego ----------
    st.markdown("""
        <style>
            .stApp {
                background-image: url("https://img.freepik.com/vector-gratis/fondo-abstracto-diseno-estilo-pixel-oscuro_1048-15775.jpg?semt=ais_hybrid&w=740");
                background-size: cover;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("❓ Adiviná el Género del Libro")

    if "juego_en_progreso" not in st.session_state:
        st.session_state.juego_en_progreso = False
    if "juego_ronda_actual" not in st.session_state:
        st.session_state.juego_ronda_actual = 0
    if "juego_puntaje" not in st.session_state:
        st.session_state.juego_puntaje = 0
    if "juego_finalizado" not in st.session_state:
        st.session_state.juego_finalizado = False

    if not st.session_state.juego_en_progreso:
        st.subheader("Seleccioná la dificultad y el generador para comenzar:")
        dificultad = st.radio("Seleccione la dificultad deseada:", ["Fácil", "Media", "Difícil"], key="juego_dificultad")

        modelo_juego = st.selectbox("Seleccione el generador a utilizar para generar las imágenes:", ["CGAN", "CVAE"], key="juego_modelo")

        if dificultad == "Fácil":
            st.session_state.juego_cant_opciones = 3
        elif dificultad == "Media":
            st.session_state.juego_cant_opciones = 5
        else:
            st.session_state.juego_cant_opciones = 7

        if st.button("🎮 Jugar"):
            st.session_state.juego_en_progreso = True
            st.session_state.juego_ronda_actual = 1
            st.session_state.juego_puntaje = 0
            st.session_state.juego_finalizado = False
            st.session_state.reiniciar_juego = True
            st.session_state.juego_modelo_seleccionado = modelo_juego
            st.rerun()
    else:
        st.markdown(f"**Ronda {st.session_state.juego_ronda_actual} de 10**")

        # Mostrar puntaje actual antes de responder
        st.markdown(f"**Puntaje actual: {st.session_state.juego_puntaje} / {st.session_state.juego_ronda_actual}**")

        if "juego_imgs" not in st.session_state or st.session_state.get("reiniciar_juego"):
            genero_correcto = random.choice(GENRES)
            genero_id = obtener_id_genero(df, genero_correcto)
            modelo = st.session_state.get("juego_modelo_seleccionado", "CGAN")
            if modelo == "CGAN":
                imgs = generar_por_genero_cgan(cGAN_generator, genero_id, 4)
            else:
                imgs = generar_por_genero_cvae(cVAE, genero_id, 4)
            imgs = normalize_individual_images(imgs)
            cantidad_opciones = st.session_state.get("juego_cant_opciones", 4)
            opciones = [genero_correcto]

            while len(opciones) < cantidad_opciones:
                gen = random.choice(GENRES)
                if gen not in opciones:
                    opciones.append(gen)
            random.shuffle(opciones)

            st.session_state.juego_imgs = imgs
            st.session_state.juego_respuesta_correcta = genero_correcto
            st.session_state.juego_opciones = opciones
            st.session_state.juego_respuesta_usuario = None
            st.session_state.juego_resultado = None
            st.session_state.reiniciar_juego = False

        st.markdown("### Imágenes generadas")
        cols = st.columns(4)
        for i, img in enumerate(st.session_state.juego_imgs):
            with cols[i % 4]:
                img = img.permute(1, 2, 0).numpy()
                st.image(img, use_container_width=True)
                st.markdown(f"<div class='image-caption'>Portada {i+1}</div>", unsafe_allow_html=True)

        respuesta = st.radio(
            "¿Cuál creés que es el género de estas portadas?",
            st.session_state.juego_opciones,
            key=f"juego_radio_{st.session_state.juego_ronda_actual}",
            disabled=st.session_state.juego_resultado is not None
        )

        if st.button("Responder") and st.session_state.juego_resultado is None:
            st.session_state.juego_respuesta_usuario = respuesta
            correcto = (respuesta == st.session_state.juego_respuesta_correcta)
            st.session_state.juego_resultado = correcto
            if correcto:
                st.session_state.juego_puntaje += 1
            st.rerun()

        if st.session_state.get("juego_resultado") is not None:
            if st.session_state.juego_resultado:
                st.success("✅ ¡Correcto!")
            else:
                st.error(f"❌ Incorrecto. La respuesta correcta era: {st.session_state.juego_respuesta_correcta}")

            # Puntaje parcial (luego de responder)
            st.markdown(f"**Puntaje actual: {st.session_state.juego_puntaje} / {st.session_state.juego_ronda_actual}**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔁 Nueva Ronda"):
                    if st.session_state.juego_ronda_actual >= 10:
                        st.session_state.juego_finalizado = True
                        st.rerun()
                    else:
                        st.session_state.juego_ronda_actual += 1
                        del st.session_state["juego_imgs"]
                        st.rerun()
            with col2:
                if st.button("🔄 Reiniciar Juego"):
                    for key in list(st.session_state.keys()):
                        if key.startswith("juego_"):
                            del st.session_state[key]
                    st.session_state.juego_en_progreso = False
                    st.session_state.juego_ronda_actual = 0
                    st.session_state.juego_puntaje = 0
                    st.rerun()

        if st.session_state.juego_finalizado:
            st.success(f"🎉 Juego terminado. Puntaje final: {st.session_state.juego_puntaje} / 10")
            if st.button("🔄 Volver a Jugar"):
                for key in list(st.session_state.keys()):
                    if key.startswith("juego_"):
                        del st.session_state[key]
                st.session_state.juego_en_progreso = False
                st.rerun()
