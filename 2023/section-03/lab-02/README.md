# Lab: AI, Representational Learning, and Word Embeddings

Welcome to an exploration of artificial intelligence (AI) where we'll dive into how computers can learn to understand words, sentences, and even entire documents. In this lab, we focus on the concept of "representational learning," which is a fancy way of saying we're teaching computers to convert complex data into a format that makes sense to them—a format that they can analyze and learn from.

## The Power of Representation

Why do we care about all this? Well, computers are great at dealing with numbers. But when it comes to things like text, they need a little help. That's where **embeddings** come in—they're like a translator, converting words into numbers (vectors) that encapsulate their meaning.

But how do we measure the meaning or the relationship between these word vectors? This is where the other pieces of our vocabulary list come into play. By using concepts like **cosine similarity**, we can get a sense of how close or far apart the meaning of words or sentences are in this numerical space.

## Vocabulary:
### 1. Embedding / Embed
- **Embedding**: An embedding is a way of converting something complicated, like a word or a picture, into a list of numbers so that a computer can understand and work with it better.
- **Embed**: To embed something means to change it into this list of numbers.

### 2. Latent
- **Latent**: Latent refers to things that are not immediately obvious or visible but are present and can be discovered through analysis. In computer science, it often means information or patterns that are hidden in data.

### 3. Vector
- **Vector**: A vector is a list of numbers. These numbers can represent anything - like the strength of different smells in a perfume, or the scores in different subjects at school. In computing, we use vectors to store and work with this kind of information.

### 4. Matrix
- **Matrix**: A matrix is like a big grid or table made up of numbers. Each row of the table can be a different vector. If vectors are lists of numbers, then a matrix is a list of lists of numbers.

### 5. Dot Product
- **Dot Product**: The dot product is a way to multiply two vectors. You multiply the matching numbers in the vectors together and then add up all those results. It's a way to find out how similar or different the vectors are.

### 6. Cosine Similarity
- **Cosine Similarity**: This is a way to measure how similar two vectors are, but unlike the dot product, it doesn't get bigger just because the numbers in the vectors are big. It's more about the direction of the vectors than their size. If two vectors point in almost the same direction, their cosine similarity is high.

### 7. Representational (as in learning)
- **Representational Learning**: This is a way of teaching a computer to understand and use data. The computer learns to convert data (like pictures, texts, or sounds) into a form (like vectors) that it can easily analyze and make decisions from.

## Practical Applications

Imagine you have a giant library of books, but no search engine. You know that some books are similar to each other, but you don't have an easy way to find them. That's the problem we're solving here—by representing words and sentences as vectors, we can build search engines for text that understand meaning, not just keywords.

## What You Will Learn

By the end of this lab, you'll learn how to:

1. **Create Embeddings**: Convert words, sentences, and paragraphs into numerical vectors.
2. **Understand Similarity**: Use mathematical tools to find the similarity between different pieces of text.
3. **Build a Search Engine**: Apply these concepts to construct a basic semantic search engine that finds relevant articles based on the meaning encoded in your search query.

This is not just about learning theory; it's about getting hands-on experience with the tools and techniques that power modern AI applications. Let's get started and see how we can teach computers to understand language a bit more like we do.
