# %% [markdown]
# # Big Picture: Clustering and Search
# We can use word embeddings to perform clustering and search tasks.
# In this notebook, we will explore how to do this using a large word list.

# %%
# !pip install sentence-transformers umap-learn plotly txtai -q --upgrade
# !wget -q https://raw.githubusercontent.com/marr75/wecodekc-scientific-computing/main/2024/lab-01/word_list.py

# %%
# Import necessary libraries
from txtai import embeddings
import word_list
import sentence_transformers
import umap
import pandas as pd
import plotly.express as px

# %%
embedding_config = {"path": "avsolatorio/GIST-small-Embedding-v0", "content": True}

# Create embeddings model
embedding_engine = embeddings.Embeddings(embedding_config)
ids = list(range(len(word_list.all_words)))
embedding_engine.index(word_list.all_words)

# %%
# Search for items related to 'royal'
embedding_engine.search("royal", limit=5)

# %%
# Search for items related to 'basketball'
embedding_engine.search("basketball", limit=5)

# %%
# Embed the word list
encoder = sentence_transformers.SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
embedded_word_list = encoder.encode(word_list.all_words)
reducer = umap.UMAP(n_components=2, random_state=42)
reduced_embedded_word_list = reducer.fit_transform(embedded_word_list)
reduced_words_df = pd.DataFrame(reduced_embedded_word_list, columns=["x", "y"])
reduced_words_df["word"] = word_list.all_words

# %%
# Plot the reduced word embeddings
fig = px.scatter(reduced_words_df, x="x", y="y", hover_data=["word"])
fig.update_layout(title="Word Embeddings Visualization", xaxis_title="X", yaxis_title="Y", legend_title="Words")
fig.update_traces(marker_size=5)
fig.show()
