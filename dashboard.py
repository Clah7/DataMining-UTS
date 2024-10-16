import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from functions import (
    plot_dominant_colors,
    preprocess_image,
    extract_features,
    initialize_centroids,
    get_dominant_colors,
    segment_image,
    show_image_with_legend,
    train_kmeans_on_dataset
)

def section_color(images):
    st.subheader("5 Warna Dominan dari Gambar yang Diunggah")
    dominant_colors = get_dominant_colors(images, num_colors=5)
    plot_dominant_colors(dominant_colors)



st.title("Dashboard Clustering Citra Udara")
# User upload gambar
st.header("Upload Gambar Anda")
uploaded_files = st.file_uploader(
    "Pilih hingga 5 gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

st.subheader("Pengaturan KMeans Clustering")
num_clusters = st.slider("Pilih jumlah cluster (1-4):", min_value=2, max_value=4, value=2)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Anda hanya dapat mengunggah maksimal 5 gambar. Harap pilih kembali.")
    else:
        st.subheader("Gambar yang Diunggah")
        images = []
        cols = st.columns(len(uploaded_files))

        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            images.append(image) 
            with cols[idx]:
                st.image(image, caption=f"Gambar {idx + 1}", use_column_width=True)
    
        # Dominant color visualization
        section_color(images)

        # Preprocess 
        preprocessed_images = [preprocess_image(np.array(image)) for image in images]

        # Ekstraksi fitur
        all_features = np.vstack([extract_features(image) for image in preprocessed_images])

        # Kluster data keseluruhan
        centroids, cluster_lab = train_kmeans_on_dataset(images, num_clusters)

        # Menampilkan hasil
        st.subheader("Hasil Segmentasi Gambar Berdasarkan Clustering")
        cols = st.columns(len(preprocessed_images))  # Create a column for each image
        for idx, (col, image) in enumerate(zip(cols, preprocessed_images)):
            segmented_image = segment_image(np.array(image), centroids)
            fig = show_image_with_legend(segmented_image, centroids, num_clusters, f"Segmentasi Gambar {idx + 1}")
            with col:
                st.pyplot(fig)
else:
    st.info("Silakan unggah hingga 5 gambar untuk melihat hasil analisis.")

