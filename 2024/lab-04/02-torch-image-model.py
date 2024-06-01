# %% [markdown]
# ## Section 2: timm/PyTorch Image Model Library
#
# ### Introduction
#
# Welcome to the second section of our lab! Today, we’ll explore timm, a library for loading pre-trained image models
# using PyTorch. Pre-trained models have already been trained on large datasets and can be used to extract useful
# features from new images. We’ll learn how to use these models to encode images into embeddings and explore some
# exciting applications.
#
# Activities
# 1. Introduction to timm
#   • We’ll start by introducing the timm library and explaining its purpose in the world of machine learning.
#   • Placeholder for URL to timm documentation.
# 2. Load a Pre-trained Model
#   • Using timm, we’ll load a pre-trained model and use it to encode a set of images into embeddings.
#   • Embeddings are compact, dense representations of images that capture their essential features.
# 3. Image Search Use Case
#   • We’ll implement an image search use case to show how embeddings can be used to find similar images.
#   • This will demonstrate the power and practicality of using pre-trained models.
# 4. 3D Visualization with UMAP
#   • We’ll use UMAP to create a 3D plot of the image embeddings.
#   • UMAP (Uniform Manifold Approximation and Projection) is a technique for dimensionality reduction that helps us
#   visualize high-dimensional data in a lower-dimensional space.
#   • We’ll investigate and discuss the clusters of images in the plot.
#
# Key Points
#
#  • timm Library: A tool for accessing pre-trained image models.
#  • Image Embeddings: Compact representations of images used for various applications.
#  • Image Search: Finding similar images using embeddings.
#  • UMAP: A technique for visualizing high-dimensional data.
#
# Let’s begin by learning about the timm library!
import base64
import io

# %%
# !pip install datasets timm umap-learn plotly pandas --upgrade

# %%
from typing import Any

import datasets
import numpy as np
import pandas as pd
import plotly.express as px
import timm.data
import torch
import umap
from IPython.display import display
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


# %%
# Utility functions for image embeddings


def get_image_embeddings(images: list[Image.Image], model: torch.nn.Module, transforms: Any) -> list[np.ndarray]:
    """
    Get the embeddings of a batch of images using a pre-trained model.

    Args:
        images (List[Image.Image]): The input images.
        model (torch.nn.Module): The pre-trained model.
        transforms: The image transformations.

    Returns:
        List[np.ndarray]: The embeddings of the images.
    """
    img_tensors = torch.stack([transforms(image) for image in images])
    with torch.no_grad():
        embeddings = model(img_tensors).detach().cpu().numpy()
    return embeddings


def reduce_embeddings(embeddings: np.ndarray, reducer: umap.UMAP, fit: bool = True) -> np.ndarray:
    """
    Reduce the dimensionality of embeddings using UMAP.

    Args:
        embeddings (np.ndarray): The input embeddings.
        reducer (umap.UMAP): The UMAP reducer.
        fit (bool): Whether to fit the reducer on the embeddings.

    Returns:
        np.ndarray: The reduced embeddings.
    """
    if fit:
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        reduced_embeddings = reducer.transform(embeddings)
    return reduced_embeddings


def pil_to_base64(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 encoded string.

    Args:
        image (Image.Image): The input image.

    Returns:
        str: The base64 encoded string of the image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def make_dataframe_from_images(dataset: datasets.Dataset) -> pd.DataFrame:
    """
    Create a DataFrame from a dataset with embeddings and reduced embeddings.

    Args:
        dataset: The dataset containing images, embeddings, and reduced embeddings.

    Returns:
        pd.DataFrame: DataFrame containing titles, images, and reduced embeddings.
    """
    df = pd.DataFrame(
        [
            {
                "label": example["label"],
                "label_name": example["label_name"],
                "image": example["b64_image"],
                "x": example["reduced_embedding"][0],
                "y": example["reduced_embedding"][1],
                "z": example["reduced_embedding"][2],
            }
            for example in dataset
        ]
    )
    return df


def plot_image_embeddings(embeddings_df: pd.DataFrame) -> None:
    """
    Plot the image embeddings DataFrame in 3D.
    """
    fig = px.scatter_3d(
        embeddings_df, x="x", y="y", z="z", hover_data=["label_name", "image"], size_max=60, template="plotly_white"
    )
    fig.update_layout(
        title="Image Embeddings Visualization",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend_title="Summaries",
    )
    fig.show()


def load_timm_model(model_name: str = "convnextv2_base.fcmae") -> (torch.nn.Module, any):
    """
    Create a TIMM model for extracting embeddings and get the appropriate transforms.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        model (torch.nn.Module): The pre-trained model.
        transforms: The transformations to apply to the images.
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()

    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False, normalize=True)

    return model, transforms


def calculate_cosine_similarity(embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a single embedding and a set of embeddings.

    Args:
        embedding (np.ndarray): The embedding of the query image.
        embeddings (np.ndarray): The embeddings of all images in the dataset.

    Returns:
        np.ndarray: Cosine similarity scores.
    """
    embedding = embedding.reshape(1, -1)  # Reshape to 2D array for cosine similarity calculation
    similarities = cosine_similarity(embedding, embeddings)
    return similarities.flatten()


def find_top_n_similar_images(query_embedding: np.ndarray, embeddings: np.ndarray, n: int = 5) -> np.ndarray:
    """
    Find the top N most similar images based on cosine similarity.

    Args:
        query_embedding (np.ndarray): The embedding of the query image.
        embeddings (np.ndarray): The embeddings of all images in the dataset.
        n (int): The number of top similar images to return.

    Returns:
        np.ndarray: Indices of the top N most similar images.
    """
    similarities = calculate_cosine_similarity(query_embedding, embeddings)
    top_n_indices = np.argsort(similarities)[-n:][::-1]  # Get indices of top N similar images, sorted by similarity
    return top_n_indices


def search_similar_images(
    query_image: Image.Image, model: torch.nn.Module, transforms: Any, dataset: datasets.Dataset, top_n: int = 5
) -> pd.DataFrame:
    """
    Search for the top N most similar images in the dataset given a query image.

    Args:
        query_image (Image.Image): The query image.
        model (torch.nn.Module): The pre-trained model.
        transforms: The image transformations.
        dataset: The dataset containing images and embeddings.
        top_n (int): The number of top similar images to return.

    Returns:
        pd.DataFrame: DataFrame containing the top N similar images and their details.
    """
    query_embedding = get_image_embeddings([query_image], model, transforms)[0]
    embeddings = np.array([example["embedding"] for example in dataset])
    top_n_indices = find_top_n_similar_images(query_embedding, embeddings, n=top_n)

    similar_images = [dataset[int(i)] for i in top_n_indices]
    df = make_dataframe_from_images(similar_images)
    return df


# %%
# Load our model and get the transforms
model, transforms = load_timm_model()
display(model, transforms)

# %%
# Load the Food101 dataset
food101 = datasets.load_dataset("food101")
# Convert food101 class labels from integer to descriptive string
label_names = food101["train"].features["label"].names
food101 = food101.map(lambda x: {"label_name": label_names[x["label"]]})
display(food101, food101["train"][0])

# %%
# Get a sample of 50 examples from the Food101 dataset
food101_sample = food101["train"].shuffle(seed=42).select(range(30))
embeddings = get_image_embeddings(food101_sample["image"], model, transforms)
reducer = umap.UMAP(n_components=3)
reduced_embeddings = reduce_embeddings(embeddings, reducer, fit=True)
embedded_and_reduced_images = food101_sample.map(
    lambda x, i: {
        "embedding": embeddings[i],
        "reduced_embedding": reduced_embeddings[i],
        "b64_image": pil_to_base64(x["image"]),
    },
    with_indices=True,
)
display(embedded_and_reduced_images[0])
df = make_dataframe_from_images(embedded_and_reduced_images)
display(df)

# %%
# Plot the image embeddings in 3D
plot_image_embeddings(df)

# %%
# Get the image you want to search for similar images
query_index = 0
query_image = food101_sample[query_index]["image"]
display(query_image)

# %%
# Search for similar images to the query image
similar_images_df = search_similar_images(query_image, model, transforms, embedded_and_reduced_images)
display(similar_images_df)

# %%
# Exercise: Find an image of your favorite food on the internet and use it as a query image to search for similar
# images in the Food101 dataset. You can download the image and load it using the PIL library. Replace the query_image
# variable with your image and run the cell to see the results.
# Define the url of the image you want to use as a query here

# Load the image using PIL

# Search for similar images to the query image

# Display the results

# Question: How well did the model perform in finding similar images to your query image?
