# %% [markdown]
# # Lab: Transfer Learning and Fine-Tuning with Sentence Transformers
#
# In this lab, we will explore transfer learning and fine-tuning techniques using pre-trained Sentence Transformers.
# Our goal is to classify Wikipedia article summaries into specific categories using a fully connected neural network
# layer added to the pre-trained model.
#
# ## Objectives
# 1. **Understand Transfer Learning**: Learn how to leverage pre-trained models for new tasks.
# 2. **Fine-Tuning Models**: Add and train new layers on top of a pre-trained model.
# 3. **Optimize Training**: Experiment with different training parameters to achieve the best performance.
# 4. **Evaluate Model**: Test the model's performance on both fitting and non-fitting topics.
#
# ## Sections
# 1. **Setup and Imports**: Import necessary libraries and modules.
# 2. **Loading Pre-trained Model**: Load a pre-trained Sentence Transformer model.
# 3. **Extending the Model**: Add a fully connected layer to the pre-trained model.
# 4. **Defining Loss and Optimizer**: Set up the loss function, optimizer, and learning rate scheduler.
# 5. **Training the Model**: Train the extended model using the provided data.
# 6. **Evaluating the Model**: Check model predictions on new topics.
# 7. **Challenge**: Optimize the training process and analyze the results.
#
# ## Instructions
# Follow the steps in each section, running the provided code and completing the challenges. Make sure to document your
# experiments, results, and analyses as you proceed.
#
# ## Setup and Imports
# Let's start by importing the necessary libraries and modules.


# %%
# fmt: off
# Allows for forward references in type hints, improving code readability and maintainability.
from __future__ import annotations
# Provides support for creating enumerations, which are a set of symbolic names bound to unique, constant values.
import enum
# Contains functions that create iterators for efficient looping, such as chain, cycle, and permutations.
import itertools
import pickle

# A deep learning framework that provides tensors and dynamic neural networks in Python with strong GPU acceleration.
import torch
# A library for easy-to-use pre-trained sentence embedding models.
import sentence_transformers
# A Python wrapper for the Wikipedia API, useful for extracting data from Wikipedia articles.
import wikipedia

# A custom module containing lists of words for different categories.
import word_list
# fmt: on

# %%
# We've used this model in the previous labs
base_model_name = "avsolatorio/GIST-small-Embedding-v0"
# Load a pre-trained Sentence Transformer model
base_model = sentence_transformers.SentenceTransformer(base_model_name)
# Get the device (typically CPU or GPU) on which the model is loaded, we'll use this device for training
device = base_model.device


# %%
# Define a new model by adding a fully connected layer
class ExtendedModel(torch.nn.Module):
    """
    A model that extends a pre-trained Sentence Transformer model with a fully connected layer.
    """

    # Init method, called when an instance of the class is created
    def __init__(self, base_model: sentence_transformers.SentenceTransformer, output_dim: int) -> None:
        """
        Initialize the model with a pre-trained Sentence Transformer model and a fully connected layer.
        """
        super(ExtendedModel, self).__init__()
        # Use the pre-trained Sentence Transformer model
        self.base_model = base_model
        # Add a fully connected layer
        self.fc = torch.nn.Linear(base_model.get_sentence_embedding_dimension(), output_dim)

    def forward(self, in_: str) -> torch.Tensor:
        """
        Forward pass of the model. Passes the input through the base model and then through the fully connected layer.
        """
        # Ensure parameters are not updated
        with torch.no_grad():
            # Get embeddings
            embeddings = self.base_model.encode(in_, convert_to_tensor=True)
        # Pass embeddings through the fully connected layer
        output = self.fc(embeddings)
        return output


# Set the output dimension; this should match the number of classes in the classification task
output_dim = 8
# Instantiate the extended model and move it to the device (CPU or GPU)
model = ExtendedModel(base_model, output_dim).to(device)
# Freeze the base model parameters to avoid updating them during training
model.base_model.requires_grad_(False)

# %%
# Let's take a look at the models
model

# %% [markdown]
# The output of the `model` shows the structure of our `ExtendedModel` class. It includes:
#
# - `base_model`: The pre-trained Sentence Transformer, which is composed of:
#   - A `Transformer` layer for encoding sentences.
#   - A `Pooling` layer for aggregating the output of the transformer model.
#   - A `Normalize` layer for normalizing the embeddings.
# - `fc`: A fully connected (`Linear`) layer that maps the sentence embeddings (with an input dimension of 384) to our
# specified output dimension (8 in this case).
#
# The parameters of the `base_model` are frozen and won't be updated during training, allowing us to focus on training
# the `fc` layer.


# %%
def train_model(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: list[str],
    train_labels: torch.Tensor,
    epochs: int = 10,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    """
    Train the model using the provided data and labels.

    Args:
        model (torch.nn.Module): The model to be trained.
        loss_function (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        train_data (torch.Tensor): The training data.
        train_labels (torch.Tensor): The training labels.
        epochs (int, optional): The number of training epochs. Defaults to 10.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None, optional): The learning rate scheduler.
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        # Reset the gradients to zero, don't want to accumulate them
        optimizer.zero_grad()
        # Forward pass: compute predictions
        predictions = model(train_data)
        # Compute the loss
        loss = loss_function(predictions, train_labels)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Update model parameters
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")  # Print training progress


# %%
class ArticleLabels(enum.Enum):
    """Enum for article labels."""

    ATHLETES = 0
    DISHES = 1
    ACTORS = 2
    MUSIC_ARTISTS = 3
    HISTORICAL_FIGURES = 4
    JOBS = 5
    COUNTRIES = 6
    COLORS = 7


# Load training data from a pickle file
training_data = [summary for _, summary in pickle.load(open("2024/lab-02/summaries.pkl", "rb"))]

# Create training labels tensor
training_labels = torch.tensor(
    list(
        # Chain the labels for each category
        itertools.chain(
            (ArticleLabels.ATHLETES.value for _ in word_list.athletes),
            (ArticleLabels.DISHES.value for _ in word_list.dishes),
            (ArticleLabels.ACTORS.value for _ in word_list.actors),
            (ArticleLabels.MUSIC_ARTISTS.value for _ in word_list.music_artists),
            (ArticleLabels.HISTORICAL_FIGURES.value for _ in word_list.historical_figures),
            (ArticleLabels.JOBS.value for _ in word_list.jobs),
            (ArticleLabels.COUNTRIES.value for _ in word_list.countries),
            (ArticleLabels.COLORS.value for _ in word_list.colors),
        )
    )
).to(device)

# %% [markdown]
# ## Challenge: Optimize the Training Process
#
# In this challenge, your goal is to achieve the best performing model possible within a fixed number of epochs (30).
# You can experiment with different optimizers, learning rates, and other parameters to see how they affect the training
# process.
#
# ### Levers You Can Use:
# - **Optimizer**: Try different optimizers such as Adam, SGD, RMSprop, etc.
# - **Learning Rate**: Adjust the learning rate to find the optimal value.
# - **Momentum**: If using SGD, experiment with different momentum values.
# - **Scheduler**: Use learning rate schedulers to adjust the learning rate during training.
#
# ### Example Optimizer Configurations:
# ```python
# # Adam optimizer
# optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.1)
#
# # SGD optimizer with momentum
# optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
#
# # RMSprop optimizer
# optimizer = torch.optim.RMSprop(model.fc.parameters(), lr=0.001, alpha=0.99)
# ```
#
# ### Example Scheduler Configurations:
# ```python
# # StepLR scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#
# # ExponentialLR scheduler
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#
# # CosineAnnealingLR scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# ```
#
# ### Your Task:
# - Experiment with different configurations to find the best performing model.
# - Document what you tried, your results, and provide a brief analysis.
#
# ### Documentation:
# 1. **What You Tried**:
#    - Optimizer: SGD with learning rate 0.01 and momentum 0.9.
#    - Learning rate scheduler: StepLR with step size 10 and gamma 0.1.
#    - Epochs: 30.
#
# 2. **Results**:
#    - Training Loss after 30 epochs: 0.2.
#
# 3. **Analysis**:
#    - The model performed better with SGD and momentum compared to Adam. The lower learning rate helped stabilize the
#      training process, resulting in a lower final training loss.
#
# Good luck, and happy training!

# %%
# Define loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()

# Optimizers: Uncomment and modify the optimizer you want to try
# SGD = Stochastic Gradient Descent, an optimization algorithm that updates the parameters based on the computed
#   gradients, the learning rate, and some randomness
# SGD with momentum, an optimization algorithm that adds a fraction of the update vector of the past time step to the
#   current update vector
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
# Adam, an optimization algorithm that is an extension of stochastic gradient descent
# optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.1)
# RMSprop, an optimization algorithm that divides the gradient by a running average of its recent magnitude
# optimizer = torch.optim.RMSprop(model.fc.parameters(), lr=0.001, alpha=0.99)

# Learning rate scheduler: Uncomment the scheduler you want to use
# StepLR, decays the learning rate by gamma every step_size epochs
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# MultiStepLR, decays the learning rate by gamma at each milestone epochs
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)
# ExponentialLR, decays the learning rate by gamma every epoch
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# CosineAnnealingLR, anneals the learning rate following the cosine function
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
# ReduceLROnPlateau, Decrease the learning rate when the loss reaches a plateau
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
scheduler = None

number_of_epochs = 30
train_model(
    model, loss_function, optimizer, training_data, training_labels, epochs=number_of_epochs, scheduler=scheduler
)

# %%


def predict(model: torch.nn.Module, input_data: str) -> torch.Tensor:
    """
    Generate predictions for the given input data.

    Args:
        model (torch.nn.Module): The trained model.
        input_data (torch.Tensor): The input data for which to generate predictions.

    Returns:
        torch.Tensor: The predicted probabilities for each class.
    """
    # Set the model to evaluation mode
    model.eval()
    # Disable gradient computation
    with torch.no_grad():
        # Forward pass: compute logits
        logits = model(input_data)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
    # Move the result to CPU and return
    return probabilities.to("cpu")


def format_predictions(probabilities: torch.Tensor, labels_enum: enum.EnumMeta) -> dict[str, float]:
    """
    Format the predicted probabilities into a readable dictionary.

    Args:
        probabilities (torch.Tensor): The predicted probabilities.
        labels_enum (enum.EnumMeta): The enumeration of label names.

    Returns:
        dict[str, float]: A dictionary mapping label names to their predicted probabilities.
    """
    probs = probabilities.squeeze().tolist()  # Convert probabilities to list
    # Map each probability to its corresponding label name
    label_probs = {labels_enum(i).name: prob for i, prob in enumerate(probs)}
    return label_probs  # Return the formatted predictions


def safe_summary(title: str) -> str | None:
    """
    Get the summary of a Wikipedia page safely
    """
    try:
        return wikipedia.summary(title, auto_suggest=False)
    except wikipedia.exceptions.PageError:
        print(f"Page '{title}' does not exist.")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Page '{title}' is ambiguous. Did you mean one of these?\ne: {e.options}")
        return None


# %% [markdown]
# ## Checking Predictions on Novel Topics
#
# In this section, we will test the model's predictions on new topics to see how well it categorizes them. We'll use
# topics that should fit into the existing categories from the `word_list` as well as topics that shouldn't.
#
# ### Examples:
# - **Topics that should fit**:
#   - **Patrick Mahomes**: As an athlete, the model should categorize him under `ATHLETES`.
#   - **Sushi**: As a type of food, the model should categorize it under `DISHES`.
#
# - **Topics that shouldn't fit**:
#   - **Quantum Computing**: This topic is unrelated to any of our categories and should ideally be classified with low
#     confidence across all categories.
#   - **Climate Change**: Similar to quantum computing, this topic doesn't fit into any of our predefined categories.
#
# ### Instructions:
# 1. Use the function `predict_and_format` to check the predictions for various topics.
# 2. Analyze the model's performance on both fitting and non-fitting topics.
# 3. Document your findings and provide a brief analysis.
#
# ### Example Code:
# ```python
# predictions = predict(model, safe_summary("Patrick Mahomes"))
# format_predictions(predictions, ArticleLabels)
#
# predictions = predict(model, safe_summary("Sushi"))
# print(predictions)
#
# predictions = predict(model, safe_summary("Quantum Computing"))
# format_predictions(predictions, ArticleLabels)
#
# predictions = predict(model, safe_summary("Climate Change"))
# format_predictions(predictions, ArticleLabels)
# ```
#
# By testing the model on these novel topics, we can gain insights into its generalization capabilities and identify
# any potential biases or limitations.

# %%
predictions = predict(model, safe_summary("Patrick Mahomes"))
format_predictions(predictions, ArticleLabels)

# %% [markdown]
# ## Summary
#
# In this lab, we explored various concepts and techniques related to transfer learning and fine-tuning models. Here’s
# a summary of what we learned:
#
# ### Transfer Learning
# - **Pre-trained Models**: We leveraged a pre-trained Sentence Transformer model to encode Wikipedia article summaries
#   into embeddings.
# - **Extending Models**: By adding a fully connected layer to the pre-trained model, we adapted it for our specific
#   classification task.
#
# ### Fine-Tuning Models
# - **Freezing Layers**: We froze the parameters of the pre-trained model to focus the training on the newly added
#   layer.
# - **Training Process**: We defined a training loop that included loss computation, backpropagation, and optimizer
#   steps.
# - **Optimizing Training**: We experimented with different optimizers, learning rates, and schedulers to achieve the
#   best performance.
#
# ### Classes and Functions
# - **Custom Classes**: We created custom classes such as `ExtendedModel` for extending the pre-trained model and
#   `ArticleLabels` for categorizing the articles.
# - **Helper Functions**: Functions like `train_model`, `predict`, and `format_predictions` helped structure our code
#   and made it reusable.
#
# ### Using Libraries
# - **PyTorch**: We used PyTorch for building and training our neural network model.
# - **Sentence Transformers**: This library provided easy-to-use pre-trained models for generating sentence embeddings.
# - **Wikipedia**: By using the Wikipedia API, we efficiently sourced article summaries for training and evaluation.
# - **Custom Utilities**: Modules like `lab_utilities` and `word_list` helped streamline our workflow by providing
#   necessary data and helper functions.
#
# ### Practical Application
# - **Novel Topics Evaluation**: We tested the model’s generalization capabilities by predicting categories for new
#   topics, both fitting and non-fitting.
# - **Documentation and Analysis**: We emphasized the importance of documenting experiments, analyzing results, and
#   understanding the impact of different training parameters.
#
# ### Challenge and Experimentation
# - **Optimizing the Model**: We challenged ourselves to find the best performing model within a fixed number of epochs
#   by experimenting with various training configurations.
#
# ## Key Takeaways
# - **Transfer Learning**: A powerful technique to adapt pre-trained models for new tasks.
# - **Model Fine-Tuning**: Essential for improving model performance on specific tasks.
# - **Structured Coding**: Writing reusable classes and functions enhances code readability and maintainability.
# - **Libraries and APIs**: Leveraging existing libraries and APIs can significantly speed up the development process.
#
# By completing this lab, you have gained hands-on experience with transfer learning, fine-tuning models, and
# optimizing training processes. These skills are valuable for developing efficient and effective machine learning
# models for a wide range of applications.
