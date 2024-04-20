# %% markdown
# # Explore Word Embeddings
# Let's explore word embeddings further. This time, we'll provide some "machinery" to make exploring them easy.
# After running the example usage, you can explore on your own by adding more words and operations.
# You will try to come up with additional sets of words that have "mathematical" relationships.

# %%
# !pip install sentence-transformers umap-learn plotly -q --upgrade

# %%
import dataclasses

import sentence_transformers
import umap
import pandas as pd
import numpy as np
import plotly.express as px

# %%
# Helpful tools
# Define the encoder, which embeds words with it's encode method
my_encoder = sentence_transformers.SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
# Define the reducer, which reduces the dimensionality of the embeddings
my_reducer = umap.UMAP(n_components=2, random_state=42)


@dataclasses.dataclass
class EmbeddedWord:
    """
    A dataclass to store the word, its embedding, and its reduced embedding
    """

    word: str
    embedding: np.ndarray
    reduced_embedding: np.ndarray
    notes: str = ""

    @staticmethod
    def from_words(words, encoder=my_encoder, reducer=my_reducer, fit=True):
        """
        Encode a list of words using the encoder and reduce their dimensionality then return a list of EmbeddedWord
        """
        embeddings = encoder.encode(words)
        if fit:
            embeddings_reduced = reducer.fit_transform(embeddings)
        else:
            embeddings_reduced = reducer.transform(embeddings)
        return [
            EmbeddedWord(word, embedding, reduced_embedding)
            for word, embedding, reduced_embedding in zip(words, embeddings, embeddings_reduced)
        ]

    def __add__(self, other):
        """
        Add two EmbeddedWord objects
        """
        return EmbeddedWord(
            f"{self.word} + {other.word}",
            self.embedding + other.embedding,
            self.reduced_embedding + other.reduced_embedding,
        )

    def __sub__(self, other):
        """
        Subtract two EmbeddedWord objects
        """
        return EmbeddedWord(
            f"{self.word} - {other.word}",
            self.embedding - other.embedding,
            self.reduced_embedding - other.reduced_embedding,
        )


def make_dataframe_from_words(embedded_words):
    """
    Create a DataFrame from a list of words
    """
    df = pd.DataFrame(
        [
            {
                "words": embedded_word.word,
                "x": embedded_word.reduced_embedding[0],
                "y": embedded_word.reduced_embedding[1],
            }
            for embedded_word in embedded_words
        ]
    )
    return df


def plot_words(words_df):
    """
    Plot the words DataFrame
    """
    fig = px.scatter(words_df, x="x", y="y", text="words", size_max=60, template="plotly_white")
    fig.update_layout(
        title="Word Embeddings Visualization",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend_title="Words",
    )
    fig.show()


def append_embedded_word_into_df(words_df, embedded_word):
    """
    Add two words and their computed embedding into the DataFrame
    """
    words_df.loc[len(words_df.index)] = [embedded_word.word, *embedded_word.reduced_embedding]


# %%
# Example usage
words = ["king", "queen", "man", "woman", "monarch"]
embedded_words = EmbeddedWord.from_words(words)
embedded_king, embedded_queen, embedded_man, embedded_woman, embedded_monarch = embedded_words
words_df = make_dataframe_from_words(embedded_words)
computed_queen = embedded_king - embedded_man + embedded_woman
append_embedded_word_into_df(words_df, computed_queen)

plot_words(words_df)

# %% markdown
# ## Explore on your own
# Now, you can explore on your own by adding more words and operations.
# Modify the code below to repeat the example code above with your new list of words and operations.
# Try to find interesting relationships between words.
# %%
# Put your words here
my_words = []
# Make embedded words from your words
my_embedded_words = _
# Make a dataframe from your embedded words
my_words_df = _
# Compute additional words
_ = _ + _ - _  # Repeat this line for each computed word
# Add the computed words into the dataframe
append_embedded_word_into_df(my_words_df, _)  # Repeat this line for each computed word
# Plot the words
_(_)
