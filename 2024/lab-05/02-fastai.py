# %% [markdown]
# # Section 2: Applying Neural Networks to Physical Laws with FastAI
#
# In the previous section, we learned the basics of PyTorch. Now, we will use FastAI, a library that simplifies training neural networks using high-level abstractions. In this section, we'll teach a neural network to understand and predict outcomes based on the Ideal Gas Law, a fundamental equation in chemistry and physics.

# %% [markdown]
# ## Environment Setup
#
# Ensure your Google Colab environment is set for this section by installing FastAI.

# %%
# !pip install fastai --upgrade

# %%
from fastai.tabular.all import TabularDataLoaders, tabular_learner, Normalize, rmse
from fastai.callback import tracker as cb_tracker
from fastai.callback.progress import ShowGraphCallback
import numpy as np
import pandas as pd
from IPython.display import display

# %% [markdown]
# ## Generating Synthetic Data for the Ideal Gas Law
#
# To apply what we've learned to real-world scenarios, we'll create synthetic data based on the Ideal Gas Law: \( PV = nRT \). Here, we treat Pressure (P), Volume (V), and Amount of substance (n) as inputs and calculate Temperature (T) as the output. This simulated dataset will help us train our model.

# %%
# Constants
R = 8.314  # Ideal Gas Constant in J/(mol*K)

# Generate random data for Pressure (P), Volume (V), and Amount of substance (n)
num_samples = 1000
P = np.random.uniform(1, 100, num_samples)  # Pressure in atmospheres
V = np.random.uniform(1, 100, num_samples)  # Volume in liters
n = np.random.uniform(0.1, 5, num_samples)  # Amount of substance in moles

# Calculate Temperature (T) using the Ideal Gas Law: T = PV / (nR)
T = (P * V) / (n * R)

# Convert data into a Pandas DataFrame
df = pd.DataFrame({"Pressure": P, "Volume": V, "Amount": n, "Temperature": T})

# %%
# Let's take a look at the first few rows and some summary stats of the DataFrame
display(df.head(), "", df.describe())

# %%
# Convert DataFrame to FastAI DataLoaders
dls = TabularDataLoaders.from_df(
    df,
    path=".",
    y_names=["Temperature"],
    cat_names=None,
    cont_names=["Pressure", "Volume", "Amount"],
    procs=[Normalize],
)

# %% [markdown]
# ## Defining the Neural Network Model
#
# Using FastAI's `tabular_learner`, we will create a neural network model. This function allows us to specify several important settings:
#
# - **y_range**: This parameter sets the range of the output values (Temperature in this case). We specify `[0,600]` because we expect our calculated temperatures to fall within this range based on the input data. Setting this range helps the model focus its learning on a specific interval of values, improving accuracy.
#
# - **layers**: This parameter defines the structure of the neural network in terms of hidden layers and the number of neurons in each layer. Here, `[200,100]` means the first hidden layer has 200 neurons, and the second has 100 neurons. More neurons and layers can model more complex relationships but may require more data and training time.
#
# - **metrics**: These are functions used to evaluate the model's performance. We use `rmse` (root mean squared error), which measures the average magnitude of the errors between predicted values and actual values. Lower values of RMSE indicate better performance, as it means the predictions are closer to the actual values.

# %%
learn = tabular_learner(dls, y_range=(0, 7000), layers=[8, 8, 8], metrics=rmse)

# %% [markdown]
# ## Training the Model
#
# Now that our model is set up, it's time to train it. Training involves showing the model the data, allowing it to make predictions, and adjusting its parameters to improve those predictions over time. We use the `fit_one_cycle` method, which is a powerful technique to speed up training and improve model performance.

# %%
# Train the model for 5 epochs using a learning rate of 0.01
# %%
# Setup learning rate scheduler
lr_scheduler = cb_tracker.ReduceLROnPlateau(factor=10, patience=3)
early_stopping = cb_tracker.EarlyStoppingCallback(patience=5)
progress_graph = ShowGraphCallback()

# Train the model for 1000 epochs, using the learning rate scheduler
epochs = 10
initial_learning_rate = 1e-1
learn.fit_one_cycle(10, initial_learning_rate, cbs=[lr_scheduler, early_stopping, progress_graph])

# %% [markdown]
# ## Evaluating the Model
#
# After training, we'll evaluate how well our model is performing by looking at its predictions on the data. FastAI provides an easy method to display the results, which includes the input values, the true outputs, and the model's predictions.

# %%
# Show results from the model's predictions, displaying actual and predicted values
learn.show_results(max_n=10)  # Display results for 10 examples

# %% [markdown]
# # Optimization Challenge
#
# Now that we've seen the baseline model in action, it's time for a challenge! Can you improve the model's performance? Here are a few areas you might consider exploring:
#
# 1. **Model Architecture**: Adjust the number of layers and the number of neurons in each layer. Does increasing or decreasing these improve performance?
# 2. **Learning Rate**: Experiment with different learning rates. What happens if you increase or decrease it?
# 3. **Epochs**: Change the number of epochs. Does training for more or fewer epochs affect the outcome?
# 4. **Learning Rate Scheduler**: Adjust the parameters of the learning rate scheduler. Can you find a better combination of `factor` and `patience`?
# 5. **Early Stopping**: Modify the `patience` of the early stopping callback. How does this change the training dynamics?
# 6. **Batch Size**: Adjust the batch size in the DataLoader. Larger or smaller batches may yield different results.
#
# Document your experiments and results. Which changes had the most significant impact and why do you think that is?

# %%
# Define the model with adjustable parameters
layers = [8, 8, 8]  # Consider changing the structure here
learning_rate = 0.01  # Experiment with different learning rates
num_epochs = 10  # Modify the number of epochs

# Create the learner
learn = tabular_learner(dls, y_range=(0, 7000), layers=layers, metrics=rmse)

# Define callbacks for learning rate scheduling and early stopping
lr_scheduler = cb_tracker.ReduceLROnPlateau(factor=10, patience=3)
early_stopping = cb_tracker.EarlyStoppingCallback(patience=5)
progress_graph = ShowGraphCallback()

# Train the model with configurable parameters
learn.fit_one_cycle(num_epochs, learning_rate, cbs=[lr_scheduler, early_stopping, progress_graph])
learn.show_results(max_n=10)  # Display results for 10 examples

# %% [markdown]
# # Bonus Challenge: Model a New Physical Equation
#
# Now that you've experimented with optimizing a neural network for the Ideal Gas Law, let's take it a step further. Can you apply what you've learned to model a different physical equation? Here are a few suggestions, but feel free to choose any other physical equation that interests you:
#
# - **Newton's Second Law of Motion**: F = ma (Force = mass x acceleration)
# - **Ohm's Law**: V = IR (Voltage = Current x Resistance)
# - **Hooke's Law**: F = kx (Force = spring constant x displacement)
#
# Your task is to:
# 1. Choose a physical equation to model.
# 2. Generate synthetic data that represents the equation.
# 3. Define and train a neural network to learn the relationship described by your chosen equation.
#
# Feel free to use resources like ChatGPT or DeepSeek's Coder to help brainstorm and code your synthetic data generation. This challenge will test your ability to apply machine learning concepts to real-world physics problems creatively.
#
# Document your process, the challenges you face, and how you overcome them. How well does your model learn the physics equation? What insights can you draw from this exercise?
