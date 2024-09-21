# %% [markdown]
# # Exploring Cross-Encoders: Your Own Application
#
# ## What is a Cross-Encoder?
#
# A cross-encoder is like a super-smart reader that can understand and compare pieces of text. It's particularly good at:
#
# 1. Figuring out how relevant a piece of text is to a question
# 2. Ranking multiple texts based on how well they answer a question
# 3. Understanding the context and meaning behind words
#
# ## Real-World Example
#
# Remember when we asked the question "Who wrote 'To Kill a Mockingbird' and what is it about?" and gave the model several statements to rank? That's a cross-encoder in action!
#
# The model was able to:
# - Identify the most relevant and accurate answer
# - Rank partial or related information appropriately
# - Understand the context of the question and the content of the book
#
# ## Why is this Important?
#
# This technology has many real-world applications:
#
# - Improving search engine results
# - Enhancing recommendation systems
# - Automating question-answering systems
# - Assisting in research and information retrieval
#
# ## Your Challenge
#
# Now it's your turn to get creative! In this notebook, you'll:
#
# 1. Practice and show your mastery of cross-encoders
# 2. Come up with your own theoretical application for this technology
# 3. Implement your idea using the cross-encoder model

# %% [markdown]
# ## First thing's first, document your idea here!

# %%
# !pip install sentence-transformers --upgrade -q

# %%
import sentence_transformers

# %%
# Load the pre-trained model
reranker = sentence_transformers.CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")

# %%
# Experiment and test your idea in the rest of this notebook
# Add code, data, and markdown cells as needed
