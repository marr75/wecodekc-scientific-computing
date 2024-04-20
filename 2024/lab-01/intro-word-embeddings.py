# %% [markdown]
# # Introduction to Word Embeddings
#
# ## Install Required Libraries
# We're going to use
# - sentence-transformers to encode words into embeddings
# - umap to reduce the dimensionality of the embeddings
# - plotly to visualize the embeddings
# So let's install them first!

# %%
# !pip install sentence-transformers umap-learn plotly -q --upgrade

# %%
import sentence_transformers
import umap
import pandas as pd
import plotly.express as px

# %%
# Load the pre-trained model
encoder = sentence_transformers.SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
# Encode a sentence
encoder.encode("Hello, World!")

# %%
# Encode a list of words
words = ["king", "queen", "man", "woman", "monarch"]
embeddings = encoder.encode(words)

# %%
# Reduce dimensions of the embeddings
reducer = umap.UMAP(n_components=2, random_state=42)
reducer.fit(embeddings)
embeddings_reduced = reducer.transform(embeddings)
embeddings_reduced

# %%
king, queen, man, woman, monarch = embeddings_reduced
# Create a DataFrame for easier plotting with Plotly
df = pd.DataFrame(embeddings_reduced, columns=["x", "y"])
df["words"] = words
df.loc[len(df.index)] = [*(king - man + woman), "queen (computed)"]

# %%
# Create a scatter plot
fig = px.scatter(df, x="x", y="y", text="words", size_max=60, template="plotly_white")

# Enhance the plot with titles and labels
fig.update_layout(
    title="Word Embeddings Visualization",
    xaxis_title="Component 1",
    yaxis_title="Component 2",
    legend_title="Words",
)

# Show the plot
fig.show()
