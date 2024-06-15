# %% [markdown]
# # Lab 5: Training a PyTorch Neural Network
#
# Welcome to Lab 5! Today, we'll learn about training a neural network using PyTorch. In this lab, we'll cover the
# basic elements of a PyTorch model, its important methods, the hyperparameters of the model, and some strategies for
# controlling training.
#
# Let's start by installing the necessary libraries.

# %% [markdown]
# ## Selecting a T4 GPU as the Runtime
#
# Before we begin, make sure you have a GPU runtime selected in Google Colab. Using a GPU will significantly speed up
# the training process. Follow these steps to select a T4 GPU:
#
# 1. **Navigate to Runtime Settings:**
#    - At the top menu, click on `Runtime`.
#    - Select `Change runtime type`.
#
# 2. **Select GPU:**
#    - In the pop-up window, find the `Hardware accelerator` section.
#    - Select the 'T4 GPU' radio button.
#
# 3. **Save Settings:**
#    - Click `Save`.
#
# Your Google Colab environment will now restart with GPU acceleration enabled.

# %% [markdown]
# ## Install Necessary Libraries
#
# We're using Google Colab, which already has most libraries installed. However, we'll need to ensure we have the
# latest versions of some important libraries.

# %%
# !pip install datasets timm plotly pandas --upgrade

# %% [markdown]
# Great! Now that we have the necessary libraries installed, let's dive into the basic elements of a PyTorch model.

# %% [markdown]
# ## Basic Elements of a PyTorch Model

# We'll start by importing the necessary libraries and defining a simple neural network model to predict the distance
# between two points using the Pythagorean theorem.

# %% [markdown]
# ### Importing Libraries

# %%
# First, let's import the libraries we'll need.
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import display


# %% [markdown]
# ### Defining the Pythagorean Theorem Model
# In PyTorch, we define a neural network by creating a class that inherits from `torch.nn.Module`. This class must
# have two methods: `__init__` and `forward`.

# %%
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
class PythagoreanModel(nn.Module):
    def __init__(self) -> None:
        """
        Initializes the layers of the neural network.
        """
        super().__init__()
        self.hidden = nn.Linear(in_features=2, out_features=12)  # Two inputs (x, y) and hidden layer with 10 neurons
        self.output = nn.Linear(in_features=12, out_features=1)  # Output layer with one neuron

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = torch.relu(self.hidden(x))  # Apply ReLU activation
        x = self.output(x)
        return x


# %% [markdown]
# In this example, we define a simple neural network with one hidden layer to predict the distance between two points.

# %% [markdown]
# ### Instantiating the Model
# Now, let's create an instance of our `PythagoreanModel` model. This is like creating our own little mathematical
# wizard who will learn to calculate the distance between two points.

# %%
model = PythagoreanModel().to(device)

# %% [markdown]
# Let's take a look at our model.

# %%
display(model)

# %% [markdown]
# ### Defining the Loss Function and Optimizer
# We need to tell our wizard how to measure its mistakes (loss function) and how to improve (optimizer).
#
# **Mean Squared Error (MSE) Loss:** This loss function measures how far off our predictions are from the actual
# distances. It takes the difference between the predicted and actual values, squares it to make sure it's positive,
# and then averages these squared differences. The smaller this number, the better our model is performing.
#
# **Adam Optimizer:** This optimizer helps our model learn and improve its predictions. It's like a smart guide that
# adjusts the model's parameters (weights) to minimize the loss. Adam is a popular choice because it's efficient and
# adapts the learning rate during training, making the learning process smoother.

# %%
loss_function = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

# %% [markdown]
# Now that we have our model, loss function, and optimizer defined, let's create some synthetic data to train our model.

# %% [markdown]
# ### Creating Synthetic Data
# Let's generate some synthetic data points to train our model. These data points will follow the Pythagorean theorem:
#
# Given coordinates (a, b), the target will be the distance c, computed as `c = sqrt(a^2 + b^2)`.
#
# Imagine we're teaching our wizard to measure the distance between two points on a map!

# %%
# Generate random (a, b) pairs
num_samples = 1000
a = np.random.uniform(low=-100, high=100, size=num_samples)  # Random values between -10 and 10
b = np.random.uniform(low=-100, high=100, size=num_samples)  # Random values between -10 and 10
c = np.sqrt(a**2 + b**2)  # Solution to the Pythagorean theorem

# Convert to PyTorch tensors
input_data = np.column_stack((a, b))  # Combine (a, b) pairs
# Our input AKA (a, b) pairs, shaped for PyTorch learning
inputs = torch.tensor(data=input_data, dtype=torch.float32).to(device)
# Our target AKA the distance, shaped for PyTorch learning
targets = torch.tensor(data=c, dtype=torch.float32).to(device).view(-1, 1)

# %% [markdown]
# Let's take a look

# %%
df = pd.DataFrame(data={"a": a, "b": b, "c": c})
fig = px.scatter_3d(df, x="a", y="b", z="c", title="Pythagorean Theorem")
fig.update_traces(marker=dict(size=3))
fig.show()

# %% [markdown]
# We've now created synthetic data following the Pythagorean theorem. Next, we will define a simple training loop to
# train our model on this data.

# %% [markdown]
# ### Training the Model
# Now that we have our data and model ready, let's define a training loop. This loop will teach our model to predict
# the distance between two points. After each step, we'll see how our model improves.

# %%
# Set the number of epochs (how many times we will go through the entire dataset)
num_epochs = 10000

# Loop over the dataset multiple times
for epoch in range(num_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()
    # Forward pass: compute predicted y by passing x to the model
    predictions = model(inputs)
    # Compute and print loss
    loss = loss_function(predictions, targets)
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # Update model parameters
    optimizer.step()
    # Print statistics every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# %% [markdown]
# Let's inspect our model's parameters and predictions.

# %%
# Inspect model parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.data.cpu().numpy()}")

# %% [markdown]
# ### Explanation of Weights and Biases
#
# The weights and biases are the parameters our model adjusts during training. We've looked at these before, but as a
# reminder:
#
# - **Weights**: These determine the influence of each input feature on the output. They are adjusted during training
# to minimize the loss.
# - **Biases**: These are additional parameters that allow the model to fit the data better by shifting the activation
# function.
#
# We'll map these values to a drawing on the board to visualize how the model uses these parameters to make predictions.

# %%
# Inspect some predictions
with torch.no_grad():
    sample_inputs = inputs[:5]
    sample_targets = targets[:5]
    sample_predictions = model(sample_inputs)
    print("Predictions from the model:")
    for i in range(5):
        input_description = f"({sample_inputs[i][0].item():.2f}, {sample_inputs[i][1].item():.2f})"
        target_description = f"{sample_targets[i].item():.2f}"
        prediction_description = f"{sample_predictions[i].item():.2f}"
        print(f"Input: {input_description}, Target: {target_description}, Prediction: {prediction_description}")

# %% [markdown]
# In this section, we've trained our model to predict distances using the Pythagorean theorem. We can see how the
# model's predictions compare to the actual distances.

# %% [markdown]
# ### Saving the Model
# Now that our model is trained, we can save it to a file so that we can load it later without retraining. This is useful if you want to reuse the model for predictions or further training.

# %%
# Save the model to a file
model_save_path = "pythagorean_model.pth"
torch.save(model.state_dict(), model_save_path)

# %% [markdown]
# ### Deleting the Model
# To demonstrate loading the model, let's first delete our current model instance.

# %%
del model

# %% [markdown]
# ### Loading the Model
# Now, we'll create a new instance of our model and load the saved parameters.

# %%
# Create a new instance of the model
loaded_model = PythagoreanModel().to(device)

# Load the saved model parameters
loaded_model.load_state_dict(torch.load(model_save_path))

# Set the model to evaluation mode
loaded_model.eval()

# %% [markdown]
# ### Verifying the Loaded Model
# To ensure our model was loaded correctly, we'll predict the same items again and check if the predictions match.

# %%
# Inspect some predictions
with torch.no_grad():
    sample_inputs = inputs[:5]
    sample_targets = targets[:5]
    sample_predictions = loaded_model(sample_inputs)
    print("Predictions from the loaded model:")
    for i in range(5):
        input_description = f"({sample_inputs[i][0].item():.2f}, {sample_inputs[i][1].item():.2f})"
        target_description = f"{sample_targets[i].item():.2f}"
        prediction_description = f"{sample_predictions[i].item():.2f}"
        print(f"Input: {input_description}, Target: {target_description}, Prediction: {prediction_description}")

# %% [markdown]
# As you can see, the predictions from the loaded model match the previous predictions, proving that we successfully
# saved and loaded the model.

# %% [markdown]
# ## Hyperparameters and Strategies for Managing Them
#
# Hyperparameters are the settings that you configure before training a model. They can significantly affect the
# model's performance and training time.
#
# Here are some important hyperparameters and strategies for managing them:
#
# - **Learning Rate**: Controls how much the model's parameters are adjusted with respect to the loss gradient. A
# smaller learning rate can make training slower but more stable, while a larger learning rate can speed up training
# but might overshoot the optimal solution.
#
# - **Batch Size**: The number of samples processed before the model is updated. Smaller batch sizes can make training
# more stable, while larger batch sizes can speed up training.
#
# - **Number of Epochs**: The number of times the entire training dataset is passed through the model. More epochs can
# lead to better performance but can also cause overfitting if too many are used.
#
# ### Smart Strategies
#
# - **Learning Rate Scheduling**: Adjusting the learning rate during training. For example, starting with a higher
# learning rate and gradually decreasing it can help the model converge faster and more accurately.
#
# - **Early Stopping**: Monitoring the model's performance on a validation set and stopping training when performance
# stops improving. This helps prevent overfitting.
#
# - **Regularization Techniques**: Methods like dropout (randomly turning off neurons during training) and weight decay
# (adding a penalty for large weights) can help the model generalize better and reduce overfitting.
#
# These strategies help manage hyperparameters effectively and improve the model's performance and training efficiency.

# %% [markdown]
# ### Learning Rate Scheduling and Early Stopping
# Now, let's add a learning rate scheduler and early stopping to our training loop.
#
# **Learning Rate Scheduler**: We'll use `ReduceLROnPlateau`, which reduces the learning rate when a metric has stopped improving. This is a smart way to adjust the learning rate during training.
#
# **Early Stopping**: We'll implement a simple early stopping mechanism that stops training if the validation loss doesn't improve for a specified number of epochs.

# %%
# Redefine model, loss function, and optimizer
smart_model = PythagoreanModel().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(params=smart_model.parameters(), lr=0.1)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=100)

# Early stopping parameters
early_stopping_patience = 1200
best_loss = float("inf")
epochs_no_improve = 0

# Set number of epochs
num_epochs = 10000

# Training loop with early stopping and learning rate scheduling
for epoch in range(num_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()
    # Forward pass: compute predicted y by passing x to the model
    predictions = smart_model(inputs)
    # Compute and print loss
    loss = loss_function(predictions, targets)
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # Update model parameters
    optimizer.step()
    # Learning rate scheduling
    scheduler.step(loss)
    # Check for early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break
    # Print statistics every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.12f}")

# %% [markdown]
# Let's inspect our model's parameters and predictions again to ensure everything is working correctly.

# %%
# Inspect model parameters
for name, param in smart_model.named_parameters():
    print(f"{name}: {param.data}")

# %% [markdown]
# ### Explanation of Weights and Biases
#
# The weights and biases are the parameters our model adjusts during training. We've looked at these before, but as a
# reminder:
#
# - **Weights**: These determine the influence of each input feature on the output. They are adjusted during training
# to minimize the loss.
# - **Biases**: These are additional parameters that allow the model to fit the data better by shifting the activation
# function.
#
# We'll map these values to a drawing on the board to visualize how the model uses these parameters to make predictions.

# %%
# Inspect some predictions
with torch.no_grad():
    sample_inputs = inputs[:5]
    sample_targets = targets[:5]
    sample_predictions = smart_model(sample_inputs)
    print("Predictions from the smart model:")
    for i in range(5):
        input_description = f"({sample_inputs[i][0].item():.2f}, {sample_inputs[i][1].item():.2f})"
        target_description = f"{sample_targets[i].item():.2f}"
        prediction_description = f"{sample_predictions[i].item():.2f}"
        print(f"Input: {input_description}, Target: {target_description}, Prediction: {prediction_description}")

# %% [markdown]
# In this section, we've added a learning rate scheduler and early stopping to our training loop. These smart
# strategies help improve the efficiency and performance of our model's training process.
