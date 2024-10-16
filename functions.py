import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import os
import streamlit as st
from PIL import Image, ImageOps, ImageDraw

def get_dominant_colors(images, num_colors=5):
    """Combine all pixels from all images and find the top dominant colors."""
    all_pixels = []

    # Collect all pixels from each image
    for image in images:
        image = image.resize((50, 50))  # Resize for faster processing
        image_array = np.array(image)
        pixels = np.reshape(image_array, (-1, 3))  # Flatten array to (n, 3)
        all_pixels.extend([tuple(pixel) for pixel in pixels])

    # Count the most common colors in the combined set of pixels
    counts = Counter(all_pixels)
    dominant_colors = counts.most_common(num_colors)
    return [color[0] for color in dominant_colors]  

def plot_dominant_colors(dominant_colors):
    """Plot the top dominant colors from the combined set of all images."""
    plt.figure(figsize=(10, 2))

    for i, color in enumerate(dominant_colors):
        plt.subplot(1, len(dominant_colors), i + 1)
        plt.imshow([[color]])
        plt.title(f"Dominant Color {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    st.pyplot(plt)

def preprocess_image(image: np.ndarray, image_size=(75, 75)) -> np.ndarray:
    """Load and preprocess the image."""
    # Jika gambar memiliki 4 channel (misalnya, PNG dengan alpha channel), ubah ke RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(image, image_size)

    # Denoise the image
    img_denoised = cv2.GaussianBlur(img, (5, 5), 0)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_denoised, -1, kernel)

    return img_sharpened

def extract_texture_features(img: np.ndarray) -> np.ndarray:
    """Extract texture features from the image."""
    gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Calculate GLCM (Gray Level Co-Occurrence Matrix) for texture features
    glcm = cv2.createGLCM(gray_img, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

    # Extract features
    contrast = cv2.calcGLCM(gray_img, "contrast")
    dissimilarity = cv2.calcGLCM(gray_img, "dissimilarity")
    homogeneity = cv2.calcGLCM(gray_img, "homogeneity")
    energy = cv2.calcGLCM(gray_img, "energy")
    correlation = cv2.calcGLCM(gray_img, "correlation")

    texture_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

    return texture_features

def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract features from the image."""
    try:
        pixels = img.reshape(-1, 3)
        combined_features = np.hstack([pixels])

        print(f"Total number of pixels/features extracted: {combined_features.shape[0]}")
        return combined_features
    except Exception as e:
        print(f"Error in extracting features: {e}")
        return np.array([])

def initialize_centroids(data, k):
    """Initialize KMeans centroids."""
    centroids = [data[np.random.randint(0, len(data))]]
    
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_centroid_index = np.searchsorted(cumulative_probabilities, r)
        centroids.append(data[next_centroid_index])

    return np.array(centroids)

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def kmeans_manual(features, k, centroids, max_iters=100):
    """Perform KMeans algorithm manually."""
    for it in range(max_iters):
        labels = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            distances = np.array([euclidean_distance(features[i], centroid) for centroid in centroids])
            labels[i] = np.argmin(distances)

        new_centroids = np.zeros(centroids.shape)
        for j in range(k):
            if np.any(labels == j):
                new_centroids[j] = features[labels == j].mean(axis=0)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
    return centroids, labels

def update_centroids(features, clusters, k):
    """Update centroids based on the mean of features in each cluster."""
    new_centroids = np.zeros((k, features.shape[1]))
    for i in range(k):
        cluster_points = features[clusters == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def has_converged(old_centroids, new_centroids, tolerance=1e-6):
    """Check if centroids have converged."""
    distances = np.linalg.norm(new_centroids - old_centroids, axis=1)
    return np.all(distances < tolerance)

def visualize_clusters(image, cluster_labels, k, centroids):
    """Create a new image based on cluster results, coloring pixels by centroid dominant color."""
    height, width, _ = image.shape
    if cluster_labels.size != height * width:
        raise ValueError(f"Ukuran cluster_labels ({cluster_labels.size}) tidak sesuai dengan dimensi gambar ({height}x{width}).")
    
    # Reshape cluster_labels
    cluster_labels_reshaped = cluster_labels.reshape(height, width)

    clustered_rgb_image = np.zeros_like(image, dtype=np.uint8)
    for cluster in range(k):
        dominant_color = centroids[cluster].astype(int)
        clustered_rgb_image[cluster_labels_reshaped == cluster] = dominant_color

    return clustered_rgb_image


def segment_image(image_array, centroids):
    """Segment image based on calculated centroids."""
    img = preprocess_image(image_array, (200, 200))  # Resize image
    features = extract_features(img) / 255.0
    
    if features.size == 0:
        raise ValueError("Tidak ada fitur yang diekstraksi dari gambar.")
    
    cluster_labels = np.zeros(features.shape[0])
    for i in range(features.shape[0]):
        distances = np.array([euclidean_distance(features[i], centroid / 255.0) for centroid in centroids])
        cluster_labels[i] = np.argmin(distances)
    
    segmented_img = visualize_clusters(img, cluster_labels, len(centroids), centroids)
    return segmented_img


def show_image_with_legend(image, centroids, k, title):
    """Display image with cluster color legend."""
    fig, ax = plt.subplots(figsize=(4, 4))  # Smaller figure size for compact display
    ax.imshow(image)
    ax.set_title(title, fontsize=10)  # Reduce font size for the title
    ax.axis('off')

    legend_labels = []
    for i in range(k):
        color_patch = plt.Rectangle((0, 0), 1, 1, fc=centroids[i] / 255.0)
        legend_labels.append(color_patch)
    
    ax.legend(legend_labels, [f'Cluster {i + 1}' for i in range(k)],
              loc="lower center", ncol=k, bbox_to_anchor=(0.5, -0.1), 
              frameon=False, borderpad=0.5, handletextpad=0.5, columnspacing=0.5)

    return fig


def train_kmeans_on_dataset(images, k):
    """Memproses semua gambar dalam direktori dan melatih KMeans."""
    # Extract features from each preprocessed image and gather them into a list
    all_features = [extract_features(preprocess_image(np.array(image))) for image in images]
    
    # Stack all features into a single numpy array for clustering
    all_features = np.vstack(all_features)
    
    # Initialize the centroids for KMeans clustering
    initial_centroids = initialize_centroids(all_features, k)
    print("Initial centroids calculated.")
    
    # Train the KMeans algorithm manually
    centroids, cluster_labels = kmeans_manual(all_features, k, initial_centroids)
    
    return centroids, cluster_labels

#mengatur image di dashboard
def make_rounded_image(image_path):
    """Membuat gambar dengan sudut melingkar."""
    img = Image.open(image_path).convert("RGB")
    # Membuat masker lingkaran
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + img.size, fill=255)
    # Terapkan masker ke gambar
    rounded_img = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    rounded_img.putalpha(mask)
    return rounded_img