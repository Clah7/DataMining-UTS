import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from functions import (
    get_dominant_color,
    plot_dominant_colors,
    preprocess_image,
    extract_features,
    initialize_centroids,
    kmeans_manual,
    train_kmeans_on_dataset,
    segment_image,
    show_image_with_legend,
)

def section_color(images):
    st.subheader("Warna Dominan dari Gambar yang Diunggah")
    plot_dominant_colors(images)



# User image upload
st.title("Dashboard Clustering Citra Udara")

# User image upload
st.header("Upload Gambar Anda")
uploaded_files = st.file_uploader(
    "Pilih hingga 5 gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Anda hanya dapat mengunggah maksimal 5 gambar. Harap pilih kembali.")
    else:
        st.subheader("Gambar yang Diunggah")
        images = []
        cols = st.columns(len(uploaded_files))

        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            images.append(image)  # Store the images in a list
            with cols[idx]:
                st.image(image, caption=f"Gambar {idx + 1}", use_column_width=True)
    
        # Dominant color visualization
        section_color(images)

        
else:
    st.info("Silakan unggah hingga 5 gambar untuk melihat hasil analisis.")

