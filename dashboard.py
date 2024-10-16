import streamlit as st
from PIL import Image
import numpy as np
import time

from functions import (
    plot_dominant_colors,
    preprocess_image,
    extract_features,
    initialize_centroids,
    get_dominant_colors,
    segment_image,
    show_image_with_legend,
    train_kmeans_on_dataset,
    make_rounded_image
)

st.title("Dashboard Kluster Citra Udara")

# Fungsi upload gambar
def upload_images(max_files):
    uploaded_files = st.file_uploader(
        f"Pilih hingga {max_files} gambar",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > max_files:
            st.warning(f"⚠️ Melebihi batas upload")
        else:
            cols = st.columns(len(uploaded_files))
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                with cols[idx]:
                    st.image(
                        image, caption=f"{uploaded_file.name}", use_column_width=True)

    return uploaded_files

def display_instructions(header, max_files):
    st.markdown(
        f"<h4 style='color: #4CAF50;'>{header}</h4>", unsafe_allow_html=True)

display_instructions("Upload Gambar Training", 10)
uploaded_training_files = upload_images(10)

display_instructions("Upload Gambar Testing", 5)
uploaded_testing_files = upload_images(5)

# Pengaturan KMeans Klustering
st.subheader("Pengaturan Klustering KMeans")
num_clusters = st.slider("Pilih jumlah cluster (2-4):",
                         min_value=2, max_value=4, value=2)

# Jika jumlah gambar training memenuhi syarat, lanjutkan proses
if uploaded_training_files and len(uploaded_training_files) <= 10:
    training_images = []
    cols = st.columns(len(uploaded_training_files))

    for idx, uploaded_file in enumerate(uploaded_training_files):
        image = Image.open(uploaded_file)
        training_images.append(image)

    # Spinner untuk preprocessing dan training
    with st.spinner("Memproses gambar training..."):
        preprocessed_training_images = [preprocess_image(
            np.array(image)) for image in training_images]
        all_training_features = np.vstack(
            [extract_features(image) for image in preprocessed_training_images])
        centroids, cluster_lab = train_kmeans_on_dataset(
            training_images, num_clusters)

    st.success("Proses training selesai!")

    # Periksa jumlah gambar testing
    if uploaded_testing_files and len(uploaded_testing_files) <= 5:
        st.subheader("Gambar Testing yang Diupload")
        testing_images = []
        cols = st.columns(len(uploaded_testing_files))

        for idx, uploaded_file in enumerate(uploaded_testing_files):
            image = Image.open(uploaded_file)
            testing_images.append(image)
            with cols[idx]:
                st.image(
                    image, caption=f"Gambar Pengujian {idx + 1}", use_column_width=True)

        # Spinner untuk segmentasi
        with st.spinner("Memproses gambar testing..."):
            preprocessed_testing_images = [preprocess_image(
                np.array(image)) for image in testing_images]

            st.subheader(
                "Hasil Segmentasi Gambar Pengujian Berdasarkan Klustering")
            cols = st.columns(len(preprocessed_testing_images))
            for idx, (col, image) in enumerate(zip(cols, preprocessed_testing_images)):
                segmented_image = segment_image(np.array(image), centroids)
                fig = show_image_with_legend(
                    segmented_image, centroids, num_clusters, f"Segmentasi Gambar Pengujian {idx + 1}")
                with col:
                    st.pyplot(fig)

        st.success("Proses pengujian selesai!")

else:
    st.info(
        "Silakan upload gambar untuk pelatihan dan pengujian untuk melihat hasil analisis.")

for _ in range(10):
    st.write("")

# Menampilkan anggota tim
st.markdown("---")  
st.markdown("""<div style="text-align: center;"><h4>Tim Pengembang</h4></div>""",
            unsafe_allow_html=True)

cols = st.columns(3)
with cols[0]:
    st.image(make_rounded_image('public/alif.jpg'), width=150,
             caption="Alif Al Husaini\n140810220036")
with cols[1]:
    st.image(make_rounded_image('public/darren.jpg'), width=150,
             caption="Darren Christian Liharja\n140810220043")
with cols[2]:
    st.image(make_rounded_image('public/jason.jpg'), width=150,
             caption="Jason Natanael Krisyanto\n140810220051")
