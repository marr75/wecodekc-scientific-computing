# %%
# !pip install sentence-transformers --upgrade -q

# %%
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import cos_sim

# %%
# Example query and documents
query = "Who wrote 'To Kill a Mockingbird' and what themes does it deal with?"
passages = [
    "To Kill a Mockingbird was written by Harper Lee.",
    "Harper Lee, an American novelist, authored the classic To Kill a Mockingbird.",
    "The author of To Kill a Mockingbird is Harper Lee, known for her vivid storytelling.",
    "Harper Lee's most famous work, To Kill a Mockingbird, remains a seminal piece of literature.",
    "Renowned author Harper Lee wrote To Kill a Mockingbird, which tackles racial issues.",
    "The book To Kill a Mockingbird, penned by Harper Lee, has been celebrated for its social impact.",
    "Harper Lee grew up in Monroeville, Alabama.",
    "In 2007, Harper Lee received the Presidential Medal of Freedom.",
    "Harper Lee was known for her reclusive nature despite her fame.",
    "Lee's second novel, published decades later, also drew significant attention.",
    "Harper Lee maintained a close friendship with fellow writer Truman Capote.",
    "Monroeville, Alabama, honors Harper Lee's legacy with an annual literary festival.",
    "To Kill a Mockingbird explores themes of racial injustice and moral growth.",
    "Set in the 1930s South, To Kill a Mockingbird follows the Finch family.",
    "The novel's central character, Scout, navigates complex social issues.",
    "Atticus Finch, a lawyer, defends a black man accused of a serious crime.",
    "To Kill a Mockingbird is widely studied in American schools for its profound themes.",
    "The courtroom scenes in To Kill a Mockingbird highlight deep-seated prejudices.",
    "The Great Gatsby captures the excesses of the Jazz Age through the eyes of Nick Carraway.",
    "Moby-Dick's narrative follows Captain Ahab's relentless pursuit of the white whale.",
    "Beloved is a powerful story about the lasting scars of slavery on an escaped slave.",
    "The Grapes of Wrath depicts the struggles of the Joad family during the Great Depression.",
    "Of Mice and Men explores the friendship and dreams of two displaced ranch workers.",
    "Catch-22 is a satirical novel highlighting the absurdities of war and bureaucracy.",
]

# %%
# Load the model
reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")

# %% [markdown]
# CrossEncoders have 2 primary methods
# ## `.predict`
# **Signature**: `predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32, show_progress_bar: bool = None) -> List[float]`
#
# **Returns**: A list of relevance scores for each sentence pair.
#
# **Usage**:
# ```python
# # Get the relevance scores for each passage
# scores = reranker.predict([(query, passage) for passage in passages])
# ```
# ## `.rank`
# **Signature**: `rank(self, query: str, documents: List[str], top_k: int = None, return_documents: bool = False) -> Union[List[int], List[str]]`
#
# **Returns**: A list of indices or documents ranked by relevance.
#
# **Usage**:
# ```python
# # Get the ranked passages
# ranked_passages = reranker.rank(query, passages, return_documents=True)
# ```

# %%
# Get the scores
reranked_results = reranker.rank(query, passages, return_documents=True)
reranked_results

# %%
# Load the embedding model to compare
embedder = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    prompts={"query": "Represent this sentence for searching relevant passages: ", "passage": ""},
)

# %%
# Get the scores from the embeddings to compare
query_embeddings = embedder.encode(query, prompt_name="query")
passage_embeddings = embedder.encode(passages, prompt_name="passage")
similarities = cos_sim(query_embeddings, passage_embeddings).tolist()[0]
mapped_results = sorted(
    [
        {"corpus_id": corpus_id, "score": similarity, "text": passage}
        for corpus_id, (similarity, passage) in enumerate(zip(similarities, passages))
    ],
    key=lambda x: x["score"],
    reverse=True,
)
mapped_results
