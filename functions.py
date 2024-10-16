import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import os
import streamlit as st

def get_dominant_color(image):
    """Get the dominant color of the image."""
    image = image.resize((50, 50))  # Resize for faster processing
    image_array = np.array(image)
    pixels = np.reshape(image_array, (-1, 3))  # Flatten array to (n, 3)

    counts = Counter([tuple(pixel) for pixel in pixels])
    dominant_color = counts.most_common(1)[0][0]
    return dominant_color

def plot_dominant_colors(images):
    """Plot the dominant colors from a list of images."""
    num_images = len(images)
    plt.figure(figsize=(10, 5))

    dominant_colors = []
    for image in images:
        dominant_color = get_dominant_color(image)
        dominant_colors.append(dominant_color)

    for i, color in enumerate(dominant_colors):
        plt.subplot(1, num_images, i + 1)
        plt.imshow([[color]]) 
        plt.title(f"Warna Dominan {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    st.pyplot(plt)

def preprocess_image(image: np.ndarray, image_size=(75, 75)) -> np.ndarray:
    """Load and preprocess the image."""
    img = cv2.resize(image, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

def train_kmeans_on_dataset(directories, k):
    """Process all images in directories and train KMeans."""
    all_features = []
    for directory in directories:
        for filename in os.listdir(directory[0]):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(directory[0], filename)
                img = cv2.imread(img_path)  # Load image with OpenCV
                if img is None:
                    continue
                img = preprocess_image(img)  # Preprocess the image
                features = extract_features(img)
                all_features.append(features)

    all_features = np.vstack(all_features)
    initial_centroids = initialize_centroids(all_features, k)
    print("Initial centroids calculated.")

    centroids, cluster_labels = kmeans_manual(all_features, k, initial_centroids)
    return centroids, cluster_labels

def visualize_clusters(image, cluster_labels, k, centroids):
    """Create a new image based on cluster results, coloring pixels by centroid dominant color."""
    clustered_rgb_image = np.zeros_like(image, dtype=np.uint8)
    height, width, _ = image.shape
    cluster_labels_reshaped = cluster_labels.reshape(height, width)

    for cluster in range(k):
        dominant_color = centroids[cluster].astype(int)
        clustered_rgb_image[cluster_labels_reshaped == cluster] = dominant_color

    return clustered_rgb_image

def segment_image(image_path, centroids):
    """Segment image based on calculated centroids."""
    img = cv2.imread(image_path)  # Load image with OpenCV
    img = preprocess_image(img)
    features = extract_features(img) / 255.0
    cluster_labels = np.zeros(features.shape[0])
    
    for i in range(features.shape[0]):
        distances = np.array([euclidean_distance(features[i], centroid / 255.0) for centroid in centroids])
        cluster_labels[i] = np.argmin(distances)
    
    segmented_img = visualize_clusters(img, cluster_labels, len(centroids), centroids)
    return segmented_img

def show_image_with_legend(image, centroids, k, title):
    """Display image with cluster color legend."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

    legend_labels = []
    for i in range(k):
        color_patch = plt.Rectangle((0, 0), 1, 1, fc=centroids[i] / 255.0)
        legend_labels.append(color_patch)
    
    plt.subplots_adjust(bottom=0.2)
    plt.legend(legend_labels, [f'Cluster {i + 1}' for i in range(k)],
               loc="lower center", ncol=k, bbox_to_anchor=(0.5, -0.1), 
               frameon=False, borderpad=1, handletextpad=1, columnspacing=1)

    plt.show()
