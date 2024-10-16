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


st.title("Dashboard Clustering Citra Udara", anchor=None)

# User upload for training images
st.header("Upload Gambar Training Anda (Maksimal 10 gambar)")
uploaded_training_files = st.file_uploader(
    "Pilih hingga 10 gambar untuk training", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# User upload for testing images
st.header("Upload Gambar Testing Anda (Maksimal 5 gambar)")
uploaded_testing_files = st.file_uploader(
    "Pilih hingga 5 gambar untuk testing", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# KMeans Clustering Settings
st.subheader("Pengaturan KMeans Clustering")
num_clusters = st.slider("Pilih jumlah cluster (1-4):", min_value=2, max_value=4, value=2)

# Check if training images are uploaded
if uploaded_training_files:
    if len(uploaded_training_files) > 10:
        st.warning("Anda hanya dapat mengunggah maksimal 10 gambar untuk training. Harap pilih kembali.")
    else:
        st.subheader("Gambar Training yang Diunggah")
        training_images = []
        cols = st.columns(len(uploaded_training_files))

        for idx, uploaded_file in enumerate(uploaded_training_files):
            image = Image.open(uploaded_file)
            training_images.append(image) 
            with cols[idx]:
                st.image(image, caption=f"Gambar Training {idx + 1}", use_column_width=True)

        # Preprocess training images
        preprocessed_training_images = [preprocess_image(np.array(image)) for image in training_images]

        # Extract features from training images
        all_training_features = np.vstack([extract_features(image) for image in preprocessed_training_images])

        # Train KMeans on training images
        centroids, cluster_lab = train_kmeans_on_dataset(training_images, num_clusters)

        # Check if testing images are uploaded
        if uploaded_testing_files:
            if len(uploaded_testing_files) > 5:
                st.warning("Anda hanya dapat mengunggah maksimal 5 gambar untuk testing. Harap pilih kembali.")
            else:
                st.subheader("Gambar Testing yang Diunggah")
                testing_images = []
                cols = st.columns(len(uploaded_testing_files))

                for idx, uploaded_file in enumerate(uploaded_testing_files):
                    image = Image.open(uploaded_file)
                    testing_images.append(image) 
                    with cols[idx]:
                        st.image(image, caption=f"Gambar Testing {idx + 1}", use_column_width=True)

                # Preprocess testing images
                preprocessed_testing_images = [preprocess_image(np.array(image)) for image in testing_images]

                # Display segmentation results for testing images
                st.subheader("Hasil Segmentasi Gambar Testing Berdasarkan Clustering")
                cols = st.columns(len(preprocessed_testing_images))  # Create a column for each testing image
                for idx, (col, image) in enumerate(zip(cols, preprocessed_testing_images)):
                    segmented_image = segment_image(np.array(image), centroids)
                    fig = show_image_with_legend(segmented_image, centroids, num_clusters, f"Segmentasi Gambar Testing {idx + 1}")
                    with col:
                        st.pyplot(fig)

else:
    st.info("Silakan unggah gambar untuk training dan testing untuk melihat hasil analisis.")
