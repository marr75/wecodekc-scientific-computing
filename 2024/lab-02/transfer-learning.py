# %% [markdown]
# # Transfer Learning: From Theory to Practice
#
# Welcome to this Jupyter notebook on Transfer Learning! This tutorial will take you on a fun, easy-to-understand journey through one of the most important topics in modern machine learning: transfer learning.
#
# ## What is Transfer Learning?
#
# Transfer Learning is a machine learning strategy where a model trained on one problem is used in some way on a second related problem. It's like teaching your computer some skills in one area and then applying these skills to a different but related task.
#
# For example, imagine you learned to play the guitar. You've never played the violin, but because of your guitar-playing skills, it's easier to learn the violin compared to someone who's never played a musical instrument before. Transfer learning works in a similar way.
#
# ## Why is it Important?
#
# In the world of machine learning, transfer learning is important for a couple of reasons:
#
# 1. **Less Data:** If you have a small amount of data for your problem, but your problem is similar to one for which a model was already trained, then transfer learning can help you by leveraging the already learned features.
#
# 2. **Less Time and Resources:** Training a deep learning model from scratch requires a lot of time, resources, and data. If a pre-trained model is used, it can save a lot of time and computational resources.
#
# ## Transfer Learning Variations
#
# There are primarily two ways we use a pre-trained model in Transfer Learning:
#
# 1. **Fine-tuning:** Here we adjust (or “fine-tune”) the weights of the pre-trained model to accomplish a new task.
#
# 2. **Feature Extraction:** Here, we treat the pre-trained model as an arbitrary feature extractor, allowing it to extract useful features from new data. We then use these features to train new layers of the network for a new task.
#
# Today, we are going to follow these steps on a pre-trained ResNet model.
#
# ## From Language Model to Chat Agent
#
# The process of evolving a language model into a chat agent or, in our case, an image classifier, can be divided into several steps. This concept is perfectly illustrated by [Andrej Karpathy's presentation](https://www.youtube.com/watch?v=bZQun8Y4L2A) on how OpenAI iteratively develops its models:
#
# 1. **Large Language Model (LLM):** We start with a large-scale model trained on a vast amount of text data from the internet. The model doesn't know any specifics about the task we're interested in, but it has learned a lot about language in general.
#
# 2. **Task-Specific Tuning:** The model is then adapted for a more specific task. Techniques such as fine-tuning or feature extraction can be employed here depending on the nature of the task. The model starts to learn more about the task using the general knowledge it acquired during the first phase.
#
# 3. **Reinforcement Learning from Human Feedback (RLHF):** Next, the model's performance is iteratively improved based on feedback from humans. Human experts review the model's outputs, whether that's generating text or classifying images, and provide feedback. The model then uses this feedback to improve.
#
# 4. **Application-Specific Tuning:** Finally, the model is further refined to fit a specific application. In the case of a chat agent, this would be fine-tuning for chat-based interaction. In our context, it would involve adapting our model to differentiate between bees and ants accurately.
#
# In this notebook, we'll be applying similar principles as we use a pre-trained ResNet model to classify images of bees and ants. We'll explore two techniques - fine-tuning and feature extraction - and observe how they impact the performance of our model in this task. Let's get started!
#
#

# %% [markdown]
# ## Import necessary libraries and modules
# torch is the State of the Art library for building deep learning models. OpenAI uses torch.

# %%
# !wget https://download.pytorch.org/tutorial/hymenoptera_data.zip

# %%
import copy
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import torchvision
import torch.utils.data
from torch import nn, optim, Tensor
from plotly import express as px
from IPython.display import display

# %%
# Define device - use GPU if available, else use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
display(device)


# %% [markdown]
# ## Here we'll define some reusable functions


# %%
def load_hymenoptera_data():
    # URL of the .zip file
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    hymenoptera_zip = current_working_directory / "hymenoptera_data.zip"

    # Send an HTTP request to the URL of the .zip file and save it to the current directory
    response = requests.get(url)
    hymenoptera_zip.write_bytes(response.content)

    # Extract the .zip file into the current directory
    with zipfile.ZipFile(hymenoptera_zip, "r") as zip_ref:
        zip_ref.extractall(current_working_directory)

    # Remove the .zip file as it's no longer needed
    hymenoptera_zip.unlink()
    data_directory = current_working_directory / "hymenoptera_data"
    return data_directory


def plot_image(input_image: Tensor, title: str = None):
    """
    Function to display an image from tensor
    """
    # Convert image to numpy
    input_image = input_image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_image = std * input_image + mean
    input_image = np.clip(input_image, 0, 1)
    fig = px.imshow(input_image)
    if title is not None:
        fig.update_layout(title=title)
    fig.show()


def train_model(model, criterion, optimizer, scheduler, max_epochs=50, max_time=120):
    """
    Function to train a model

    Arguments:
    model : The model to be trained
    criterion : The loss function
    optimizer : The optimization function
    scheduler : Learning rate scheduler for adjusting the learning rate during training
    num_epochs : Number of epochs for training the model (default is 25)

    Returns:
    model : The trained model with the best validation accuracy
    """
    start_time = time.time()
    elapsed_time = 0
    epoch = 1

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    while epoch <= max_epochs and elapsed_time <= max_time:
        print(f"Epoch {epoch} / {max_epochs}, {elapsed_time}s / {max_time}s")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
            if phase == "train":
                scheduler.step()

            # Calculate the epoch's average loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Keep a copy of the best performing model
            if phase == "val" and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        epoch += 1
        elapsed_time = round(time.time() - start_time, ndigits=1)
        print()

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_accuracy:4f}")

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


def visualize_model_predictions(model, batches=1):
    """
    Visualize model predictions for a specific number of images.

    Args:
        model (nn.Module): The trained model.
        number_of_images (int, optional): Number of images to display. Default is 6.
    """
    was_training = model.training
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            # Make a grid from the images
            image_grid = torchvision.utils.make_grid(inputs.cpu().data)

            # Get the class names for each prediction
            predicted_class_names = [class_names[x] for x in predictions]

            # Show the images with predicted class names as titles
            plot_image(image_grid, title=", ".join(predicted_class_names))

            batches -= 1
            if batches <= 0:
                break  # We only need one batch

    model.train(mode=was_training)


# %% [markdown] id="e8256f3d"
# ## Download and prepare the data
#
# We're going to use just pictures of bees and ants from a popular AI dataset with hundreds of thousands of photos

# %% id="fda7233b"
# Prepare the dataset
data_transformations = {
    # Data augmentation and normalization for training
    "train": torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    # Just normalization for validation (no flipping, no random resizing!)
    "val": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

current_working_directory = Path.cwd()

data_directory = load_hymenoptera_data()

image_datasets = {
    x: torchvision.datasets.ImageFolder(os.path.join(data_directory, x), data_transformations[x])
    for x in ["train", "val"]
}
data_loaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

display(
    f"Dataset dimensions: {dataset_sizes}",
    f"Class names: {class_names}",
)

# %% id="4695db61"
# Get a batch of training data
inputs, classes = next(iter(data_loaders["train"]))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

plot_image(out, title=", ".join([class_names[x] for x in classes]))

# %% [markdown] id="afbe1ee1"
# ## Loading the pre-trained model
#
# ...and replacing the final layer to repurpose the pre-trained model to detect bees and ants!

# %% id="727a526a"
# Fine-tuning the convolutional neural network
# Load a pretrained model and reset final fully connected layer.
model_to_finetune = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

# We will create a new fully connected layer
# The original model ended with a fully-connected layer called "fc", it's located at `.fc`
# We'll use the same number inputs and give it only 2 outputs (for ants and bees)
n_input_features = model_to_finetune.fc.in_features
n_output_features = len(class_names)
model_to_finetune.fc = nn.Linear(n_input_features, n_output_features)

# Optimize the model for use with our device
model_to_finetune = model_to_finetune.to(device)

# Note that all parameters are being optimized
optimizer_finetune = optim.SGD(model_to_finetune.parameters(), lr=0.001, momentum=0.9)

# Decay learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_finetune, step_size=7, gamma=0.1)

# Cross entropy loss will be the loss criterion
criterion = nn.CrossEntropyLoss()

# Train and evaluate the fine-tuned model
model_to_finetune = train_model(
    model_to_finetune, criterion, optimizer_finetune, exp_lr_scheduler, max_epochs=50, max_time=120
)

# %% [markdown] id="00140e24"
# ## Now we'll show the model's predictions

# %% id="e0af4e4f"
visualize_model_predictions(model_to_finetune, batches=3)

# %% [markdown] id="zYZXQYeVLGzp"
# # Pre-trained model as fixed feature extractor

# %% id="40d9a5ae"
# Load a pretrained model
model_with_ffe = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# Freeze all the parameters
for param in model_with_ffe.parameters():
    param.requires_grad = False

# Again, we will create a new fully connected layer
# The code is identical but this time, this layer will be the only layer optimized
n_input_features = model_with_ffe.fc.in_features
n_output_features = len(class_names)
model_with_ffe.fc = nn.Linear(n_input_features, n_output_features)
model_with_ffe = model_with_ffe.to(device)

# Optimize the model for use with our device
model_with_ffe = model_with_ffe.to(device)

criterion = nn.CrossEntropyLoss()

# Only the final, fully-connected layer is being optimized.
optimizer_ffe = optim.SGD(model_with_ffe.fc.parameters(), lr=0.001, momentum=0.9)

# Decay learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ffe, step_size=7, gamma=0.1)

# Train and evaluate the fine-tuned model
model_with_ffe = train_model(model_with_ffe, criterion, optimizer_ffe, exp_lr_scheduler, max_epochs=50, max_time=120)

# %% id="cgPXBEG4LGzp"
visualize_model_predictions(model_with_ffe, batches=3)

# %% id="ZqdJU7m2MBhO"
