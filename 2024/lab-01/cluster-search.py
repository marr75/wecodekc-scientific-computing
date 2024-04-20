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
