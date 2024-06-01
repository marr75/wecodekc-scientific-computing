# %% [markdown]
# Section 1: Datasets
#
# Introduction
#
# Welcome to the first section of our lab! Today, we’ll learn about datasets in the Hugging Face ecosystem. Datasets
# are collections of data that we can use to train and evaluate machine learning models. Hugging Face provides an easy
# way to access and use a wide variety of datasets.
#
# Activities
#
#  1. Explore Dataset Pages on Hugging Face
# 	 - We’ll start by looking at a few dataset pages on the Hugging Face website. This will help us understand what
# 	 kind of datasets are available and how they are structured.
#    - [Rotten Tomato](https://huggingface.co/datasets/rotten_tomatoes)
# 	 - [IMDb](https://huggingface.co/datasets/imdb)
# 	 - [SQuAD](https://huggingface.co/datasets/squad)
# 	 - [Tatoeba](https://huggingface.co/datasets/tatoeba)
# 	 - [Wikipedia](https://huggingface.co/datasets/wikipedia
#    - [American Stories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
# 	 - [Food101](https://huggingface.co/datasets/food101)
# 	 - [Wikiart](https://huggingface.co/datasets/huggan/wikiart)
# 	 - [Wikiart Faces](https://huggingface.co/datasets/asahi417/wikiart-face)
#    - [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)
# 	 - [A Complete Guide to Audio Datasets](https://huggingface.co/blog/audio-datasets)
#  2. Use the Dataset Viewer
# 	- Hugging Face has a dataset viewer that allows us to browse through the data without any coding. This is a
# 	great way to get a quick overview of the dataset.
# 	- Placeholder for URL to dataset viewer.
#  3. Load a Portion of a Dataset in Colab
# 	 - We’ll hop into a Colab notebook to load a portion of a dataset. This will give us hands-on experience with
# 	 using the Hugging Face datasets library.
# 	 - We’ll look at some features like slicing, filtering, and shuffling the data.
#
# Key Points
#
# 	•	Hugging Face Datasets Library: A powerful tool for accessing and using various datasets.
# 	•	Dataset Pages: Provide information about each dataset, including descriptions and sample data.
# 	•	Dataset Viewer: An interactive tool to explore datasets.
# 	•	Colab Notebook: A practical environment to load and manipulate datasets.
#
# Let’s get started by exploring the Hugging Face dataset pages!

# %%
# !pip install datasets --upgrade

# %%
import datasets

from IPython.display import display

# %%
# Working with text datasets
# Load the Rotten Tomato dataset
rotten_tomatoes = datasets.load_dataset("rotten_tomatoes")
# Display the dataset along with the first example
display(rotten_tomatoes, rotten_tomatoes["train"][0])

# %%
# Some datasets have "idiosyncrasies" that make them unique. For example, the American Stories dataset has a
# "subset_years" argument that allows us to filter the dataset by year.
# Load the American Stories dataset
american_stories = datasets.load_dataset(
    "dell-research-harvard/AmericanStories", "subset_years", year_list=["1809", "1810"], trust_remote_code=True
)
# Display the dataset along with the first example
display(american_stories, american_stories["1809"][:3])

# %%
# Working with image datasets
# Get a sample of 50 examples from the Food101 dataset
food101_sample_50 = datasets.load_dataset("food101", split="train[:50]")
# Get a sample of 1% of the examples from the Food101 dataset
food101_sample_1_percent = datasets.load_dataset("food101", split="train[:1%]")
# Display the samples, along with the last example in each sample set
display(
    food101_sample_50,
    food101_sample_50[-1],
    food101_sample_1_percent,
    food101_sample_1_percent[-1],
)

# %%
# Working with audio datasets
# Audio datasets tend to be LARGE so your instructor will load the People's Speech dataset as a demonstration
# Load the GigaSpeech dataset
# import dotenv; dotenv.load_dotenv()
# gigaspeech = datasets.load_dataset("speechcolab/gigaspeech", "dev")
# # Display the dataset along with the first example
# display(gigaspeech, gigaspeech["train"][0])

# %%
# Try it out for yourself
# Load a dataset of your choice and display the first example(s)
# Ask your instructor for suggestions or feel free to explore the Hugging Face website for inspiration!
# If you have problems, copy paste the error into an AI assistant and see if it can help you and if not, ask your
# instructor for help.
# Replace "______" with the name of the dataset you want to load
my_dataset = datasets.load_dataset("______")
# Display the dataset along with the first example
# Replace the blanks with code to display the dataset and the first example
_____(_____, _____[0])
