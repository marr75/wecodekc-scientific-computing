# %%
# !pip install sentence-transformers --upgrade -q

# %%
import sentence_transformers

# %%
# Load the pre-trained model
reranker = sentence_transformers.CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

# %%
# Examine the internals of the model
# The "model" of a cross encoder is on the "model" attribute, create a variable for it here


# The tokenizer is on the "tokenizer" attribute, create a variable for it here


# %%
# Let's see the model's architecture, in a jupyter notebook, you can use the variable name to display the model


# %%
# Let's see the tokenizer's architecture, in a jupyter notebook, you can use the variable name to display the tokenizer
