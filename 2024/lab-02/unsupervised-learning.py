# %% [markdown]
# # Introduction to Unsupervised Learning
# In this notebook, we will explore how to use unsupervised learning algorithms to cluster and explore data.
# Unsupervised learning is a type of machine learning that involves training models on data that does not have any
# labels. The goal of unsupervised learning is to find patterns and relationships in the data without any prior
# knowledge of the output. Unsupervised learning is often cheaper and easier to implement than supervised learning
# because it does not require labeled data.

# %% [markdown]
# ## Pre-questions
# - Can you think of some reasons why unsupervised learning could be powerful?
# - What could you do with unsupervised learning that you couldn't do with supervised learning?
# - What are some possible applications of unsupervised learning?

# %% [markdown]
# ## Installing Libraries
# We will need the following libraries for this notebook:
# - sentence-transformers: A library for computing sentence embeddings
# - umap-learn: A library for dimensionality reduction
# - plotly: A library for creating interactive plots

# %%
# !pip install sentence-transformers umap-learn plotly tqdm wikipedia -q --upgrade
# !wget -q https://raw.githubusercontent.com/marr75/wecodekc-scientific-computing/main/2024/lab-02/lab_utilities.py
# !wget -q https://raw.githubusercontent.com/marr75/wecodekc-scientific-computing/main/2024/lab-02/word_list.py
# !wget -q https://raw.githubusercontent.com/marr75/wecodekc-scientific-computing/main/2024/lab-02/summaries.pkl

# %%
import wikipedia
from sklearn.cluster import KMeans
from IPython.display import display

from lab_utilities import (
    EmbeddedText,
    safe_summary,
    make_dataframe_from_text,
    plot_embeddings,
    get_summaries_from_word_list,
    save_emergency_summaries,
)
import word_list

# %% [markdown]
# ## Data Source: Wikipedia
# We will use the Wikipedia API to get the summary of a topic. Let's start by getting the summary of the topic "Transfer
# learning". We will next grab hundreds of summaries from wikipedia and use them to explore unsupervised learning.

# %%
summary = wikipedia.summary("Transfer learning")
display(summary)

# %% [markdown]
# ## Warm-up: Add a couple of your own topics
# Add a couple of your own topics to the word list and get their summaries. You can use the `safe_summary` function to
# get the summary of a topic. If the summary is not available, the function will print a helpful message and return
# `None`.

# %%
# Add your own topics here. Use these example method calls to get the summaries.
display(wikipedia.search("Mario Kart"))
display(wikipedia.summary("Mario Kart", auto_suggest=False))
my_summaries = [
    # ("Mario Kart", safe_summary("Mario Kart")),
]

# %%
summaries = get_summaries_from_word_list(word_list.all_words, emergency=True)
titles = [title for title, _ in summaries]
summaries = [summary for _, summary in summaries]
display(summaries[:5])

# %% [markdown]
# ## Dimensionality Reduction (UMAP)
# Dimensionality reduction is a technique used to reduce the number of dimensions in a dataset while preserving the
# structure of the data. In this section, we will use the UMAP algorithm to reduce the dimensionality of the embeddings
# of the summaries we collected from Wikipedia.

# %%
embedded_words = EmbeddedText.from_text(titles, summaries)
display(embedded_words[0])
embedded_df = make_dataframe_from_text(embedded_words)

# %%
plot_embeddings(embedded_df)

# %% [markdown]
# ## Check our understanding: Visualizing Word Embeddings in 2D
# The plot above shows the word embeddings of the summaries we collected from Wikipedia. Each point represents a
# summary, and the distance between points represents the similarity between the summaries.
# Questions:
# - What do you observe from the plot?
# - Are there any clusters of summaries?
# - Are there any summaries that are far apart from the rest?

# %% [markdown]
# ## Clustering
# Clustering is a type of unsupervised learning that involves grouping similar data points together. In this section,
# we will use the KMeans algorithm to cluster the summaries we collected from Wikipedia.

# %%
kmeans = KMeans(n_clusters=8, random_state=42)
embedded_df["cluster"] = kmeans.fit_predict(embedded_df[["x", "y"]])
plot_embeddings(embedded_df, with_cluster=True)

# %% [markdown]
# ## Check our understanding: Clustering
# Questions:
# - Are the clusters meaningful?
# - How many clusters do you see?
# - What are some of the topics in each cluster?
# - Did we do anything to let the computer know what the documents were about?
# - How could this be useful?
