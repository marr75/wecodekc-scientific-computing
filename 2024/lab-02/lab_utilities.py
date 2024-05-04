from __future__ import annotations
import dataclasses

import numpy as np
import pandas as pd
import plotly.express as px
import sentence_transformers
import umap
import wikipedia
from tqdm import tqdm

# Define the encoder, which embeds words with it's encode method
my_encoder = sentence_transformers.SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
# Define the reducer, which reduces the dimensionality of the embeddings
my_reducer = umap.UMAP(n_components=2, random_state=42)


@dataclasses.dataclass
class EmbeddedText:
    """
    A dataclass to store the text, its embedding, and its reduced embedding
    """

    title: str
    text: str
    embedding: np.ndarray
    reduced_embedding: np.ndarray
    notes: str = ""

    @classmethod
    def from_text(
        cls,
        titles: list[str],
        texts: list[str],
        encoder: sentence_transformers.SentenceTransformer = my_encoder,
        reducer: umap.UMAP = my_reducer,
        fit: bool = True,
    ) -> list[EmbeddedText]:
        """
        Encode a list of words using the encoder and reduce their dimensionality then return a list of EmbeddedText
        """
        embeddings = encoder.encode(texts)
        if fit:
            embeddings_reduced = reducer.fit_transform(embeddings)
        else:
            embeddings_reduced = reducer.transform(embeddings)
        return [
            cls(title, text, embedding, reduced_embedding)
            for title, text, embedding, reduced_embedding in zip(titles, texts, embeddings, embeddings_reduced)
        ]

    def __add__(self, other: EmbeddedText) -> EmbeddedText:
        """
        Add two EmbeddedText objects
        """
        return EmbeddedText(
            f"{self.title} + {other.title}",
            f"{self.text} + {other.text}",
            self.embedding + other.embedding,
            self.reduced_embedding + other.reduced_embedding,
        )

    def __sub__(self, other: EmbeddedText) -> EmbeddedText:
        """
        Subtract two EmbeddedText objects
        """
        return EmbeddedText(
            f"{self.title} - {other.title}",
            f"{self.text} - {other.text}",
            self.embedding - other.embedding,
            self.reduced_embedding - other.reduced_embedding,
        )


def make_dataframe_from_text(embedded_items: list[EmbeddedText]) -> pd.DataFrame:
    """
    Create a DataFrame from a list of embeddings
    """
    df = pd.DataFrame(
        [
            {
                "title": embedded_text.title,
                "document": embedded_text.text,
                "x": embedded_text.reduced_embedding[0],
                "y": embedded_text.reduced_embedding[1],
            }
            for embedded_text in embedded_items
        ]
    )
    return df


def plot_embeddings(embeddings_df: pd.DataFrame, with_cluster: bool = False) -> None:
    """
    Plot the words DataFrame
    """
    fig = px.scatter(embeddings_df, x="x", y="y", text="title", size_max=60, template="plotly_white")
    if with_cluster:
        fig.update_traces(marker=dict(color=embeddings_df["cluster"]))
    fig.update_layout(
        title="Word Embeddings Visualization",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend_title="Summaries",
    )
    fig.show()


def append_embedded_word_into_df(embedding_df: pd.DataFrame, embedded_text: EmbeddedText) -> None:
    """
    Add two words and their computed embedding into the DataFrame
    """
    embedding_df.loc[len(embedding_df.index)] = [
        embedded_text.title,
        embedded_text.text,
        *embedded_text.reduced_embedding,
    ]


def safe_summary(title: str) -> tuple[str, str] | None:
    """
    Get the summary of a Wikipedia page safely
    """
    try:
        return title, wikipedia.summary(title, auto_suggest=False)
    except wikipedia.exceptions.PageError:
        print(f"Page '{title}' does not exist.")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Page '{title}' is ambiguous. Did you mean one of these?\ne: {e.options}")
        return None


def get_summaries_from_word_list(word_list: list[str], emergency: bool = False) -> list[tuple[str, str]]:
    """
    Get summaries from a list of words
    """
    if emergency:
        return emergency_load_summaries()
    wrapped_words = tqdm(word_list)
    summaries = (safe_summary(word) for word in wrapped_words)
    filtered = [(title, summary) for title, summary in summaries if summary is not None]
    return filtered


def emergency_load_summaries(path: str = "summaries.pkl") -> list[tuple[str, str]]:
    """
    Load summaries from a pickle file (in case downloading them fails)
    """
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def save_emergency_summaries(summaries: list[tuple[str, str]], path: str = "summaries.pkl") -> None:
    """
    Save summaries to a pickle file
    """
    import pickle

    with open(path, "wb") as f:
        pickle.dump(summaries, f)
