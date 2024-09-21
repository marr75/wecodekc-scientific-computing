# %% [markdown]
# # Visualizing Word Embeddings in English and Spanish
#
# In this notebook, we'll load two lists of the most common English and Spanish words, generate word embeddings, reduce
# their dimensionality, and visualize the results in a scatter plot.

# %% [markdown]
# ## Install Required Libraries
# We'll be using:
# - `sentence-transformers` to encode words into embeddings
# - `umap-learn` to reduce the dimensionality of the embeddings
# - `plotly` to visualize the embeddings

# %%
# !pip install sentence-transformers umap-learn plotly -q --upgrade

# %%
import sentence_transformers
import umap
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load Word Lists
# We'll load two files: one for English words and one for Spanish words.

# %%
# Load English and Spanish words from text files
with open("english-words.txt") as f:
    english_words = [line.strip() for line in f.readlines()]

with open("spanish-words.txt") as f:
    spanish_words = [line.strip() for line in f.readlines()]

# Combine English and Spanish words for embedding
words = english_words + spanish_words

# %% [markdown]
# ## Generate Word Embeddings
# We'll use the pre-trained model `avsolatorio/GIST-small-Embedding-v0` from Hugging Face to generate embeddings for our words.

# %%
# Load the pre-trained model
encoder = sentence_transformers.SentenceTransformer(
    "intfloat/multilingual-e5-small", prompts={"query": "query: "}, default_prompt_name="query"
)

# Encode the list of words
embeddings = encoder.encode(words)

# %% [markdown]
# ## Dimensionality Reduction
# We'll use UMAP to reduce the dimensionality of the embeddings to 2D so that we can plot them.

# %%
# Reduce dimensions of the embeddings
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_reduced = reducer.fit_transform(embeddings)

# %% [markdown]
# ## Create DataFrame for Visualization
# We'll label the words as either English or Spanish and store their reduced embeddings in a DataFrame.

# %%
# Create a DataFrame for easier plotting
df = pd.DataFrame(embeddings_reduced, columns=["x", "y"])
df["language"] = ["English"] * len(english_words) + ["Spanish"] * len(spanish_words)
df["words"] = words

# %% [markdown]
# ## Visualization
# We'll create a scatter plot using Plotly, coloring the points based on language.

# %%
# Create a scatter plot
fig = px.scatter(df, x="x", y="y", text="words", color="language", size_max=60, template="plotly_white")

# Enhance the plot with titles and labels
fig.update_layout(
    title="Word Embeddings Visualization (English vs. Spanish)",
    xaxis_title="Component 1",
    yaxis_title="Component 2",
    legend_title="Language",
)

# %%
# Show the plot
fig.show()

# %% [markdown]
# ## Separate KDE Plots for English and Spanish Embeddings
# We'll generate separate KDE plots for each language.

# %%
# Set up the matplotlib figure for English words
plt.figure(figsize=(10, 8))

# Plot KDE for English words
sns.kdeplot(
    x=df[df["language"] == "English"]["x"],
    y=df[df["language"] == "English"]["y"],
    cmap="Blues",
    shade=True,
    bw_adjust=0.5,
)
# Enhance the plot for English words
plt.title("KDE Plot for English Word Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)

# Show the English plot
plt.show()

# %% [markdown]
# Now, we'll create a separate KDE plot for Spanish words.

# %%
# Set up the matplotlib figure for Spanish words
plt.figure(figsize=(10, 8))

# Plot KDE for Spanish words
sns.kdeplot(
    x=df[df["language"] == "Spanish"]["x"],
    y=df[df["language"] == "Spanish"]["y"],
    cmap="Reds",
    shade=True,
    bw_adjust=0.5,
)

# Enhance the plot for Spanish words
plt.title("KDE Plot for Spanish Word Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)

# Show the Spanish plot
plt.show()
