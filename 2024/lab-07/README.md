# Large Language Model Review

## Introduction


## Warm-up: How AI is helping us talk to Animals

Based on the content of your slide deck for Lab 7, here’s a Markdown README that you can use for the lab:

---

# WeCode KC - Scientific Computing & AI Lab 7: Large Language Model Review

## Overview

In this lab, we will explore the world of **Large Language Models (LLMs)** and how AI is advancing fields such as 
communication with animals, software development, and ethical decision-making. We will use Python-based AI tools to 
experiment with word embeddings and dimensionality reduction techniques to visualize multilingual word data.

### Topics Covered
- **AI and Communication with Animals**
- **Significance of AI across various domains**
- **How Large Language Models (LLMs) Work**
- **Key Techniques used in LLMs**
- **Ethical Considerations**

---

## Lab Outline

### 1. Warm-Up: How AI Is Helping Us Talk to Animals
We'll begin with a short discussion on AI's role in animal communication, prompted by a video. Consider these questions:
- Do you recognize any concepts and techniques from the video?
- What tools and technologies do we already have to work on this problem?
- What are we missing to take this further?

### 2. Significance of AI
AI is impacting many areas of our lives. We will break this down into four main categories:
- **Scientific**: How AI is enabling breakthroughs in various fields.
- **Economic**: AI’s role in software development and automation.
- **Artistic & Cultural**: AI’s influence on creative fields like writing and art.
- **Ethical**: What responsibilities do we have as we advance AI?

### 3. How Do Large Language Models Work?
A review of neural networks and transformers will help us understand how LLMs:
- **Store and retrieve information**: How do LLMs like GPT store "facts"?
- **Understanding vs. processing**: Do humans really understand how LLMs work?
- We'll discuss both sides of the argument and ask you to form your own opinion.

### 4. Techniques in LLMs
In this section, we will look at the cutting-edge methods that improve LLM performance:
- **RAG (Retrieval Augmented Generation)**: Combining retrieval systems with generation models.
- **PAL (Program Aided Logic)**: Integrating external logic or code into language models.
- **CoT (Chain of Thought)**: Enhancing reasoning in multi-step problems.
- **Agentic AI**: Using tools and multiple passes to break down complex tasks.

### 5. Ethical Considerations
The ethics of generative AI are critical. We'll explore questions such as:
- What types of harm can AI cause, and who is most vulnerable?
- What can we do to minimize harm and ensure fairness?
- What decisions should humans be in charge of, even with advanced AI?

---

## Learning Goals

By the end of this lab, students should be able to:
1. Understand the significance of AI across multiple domains.
2. Grasp the basic working mechanisms of Large Language Models.
3. Familiarize themselves with techniques such as RAG, PAL, CoT, and agentic AI.
4. Engage in discussions on the ethical implications of generative AI.

---

## Lab Activity 1

[Warm Up](01-warm-up.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marr75/wecodekc-scientific-computing/blob/main/2024/lab-07/01-warm-up.ipynb)

Students will be guided through a coding exercise where they will:
1. Encode a set of English and Spanish words using pre-trained word embedding models.
2. Visualize these embeddings using UMAP dimensionality reduction.
3. Create Kernel Density Estimate (KDE) plots for both languages, showing word distributions in 2D space.

---

## Lab Activity 2

## Exercise: Evaluating LLM Performance with and without RAG/PAL

### Overview
In this exercise, we will demonstrate the difference between a standard LLM and one enhanced with PAL (Program-Aided 
Logic) or RAG (Retrieval-Augmented Generation). The task involves solving a series of difficult floating-point math 
problems that the LLM is not typically trained to handle, and comparing results between the two approaches.

### Objective
Students will:
- See how LLMs struggle with "out-of-distribution" problems (such as floating-point arithmetic) without external tools.
- Use RAG/PAL to significantly improve the LLM’s accuracy by feeding it the problem-answer pairs as reference material.
- Compare and reflect on the effectiveness of PAL and RAG for improving LLM performance.

### Steps

#### 1. **Baseline Test (No PAL/RAG)**
   - Present the LLM with 5 randomly chosen math problems from the list of generated floating-point problems.
   - Do **not** provide the LLM with any external resources (no PAL or RAG) and ask it to solve the problems on its own.
   - Record the results and observe how accurately the LLM can compute the answers. 

#### 2. **Enhanced Test (With PAL/RAG)**
   - Now, provide the LLM with access to the entire list of problem-answer pairs.
   - Ask the LLM to solve the same problems, but this time using **PAL** or **RAG** to retrieve the correct answers 
   - from the reference data.
   - Record the results and observe the improvement in accuracy.

#### 3. **Comparison and Analysis**
   - Compare the results of the **baseline test** (without PAL/RAG) with the **enhanced test** (with PAL/RAG).
   - Discuss:
     - How much more accurate the LLM is when it uses RAG/PAL.
     - Why LLMs struggle with out-of-distribution tasks like floating-point arithmetic.
     - How PAL and RAG can augment the capabilities of LLMs for specific use cases.

### Materials
- The list of 30 generated floating-point problems and their answers (`floating_point_problems_with_answers.txt`).
- LLM for testing both without and with PAL/RAG.
  
### Key Learning Points
- **RAG**: How retrieval-augmented generation helps LLMs access and use external information to improve accuracy.
- **PAL**: How program-aided logic enhances LLMs by offloading complex tasks like math to external code execution.
- **LLM Limitations**: Understanding the limitations of LLMs when dealing with complex, uncommon, or 
- out-of-training-distribution tasks.
  
This hands-on demonstration helps students understand both the weaknesses of large language models and the powerful 
solutions PAL and RAG provide in practice.

---

## Ethical Reflection

In addition to technical skills, the lab encourages students to reflect on how their knowledge of AI and LLMs could be 
applied responsibly in real-world scenarios.

## Lab Activity 3

[Warm Up](03-warm-up.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marr75/wecodekc-scientific-computing/blob/main/2024/lab-07/03-warm-up.ipynb)
[DeBERTa Model Explanation](deberta-model-explanation.md)

Students will examine the structure of a State of the Art Cross-Encoder model

## Lab Activity 4

[Cross Encoder Example](04-cross-encoder-example.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marr75/wecodekc-scientific-computing/blob/main/2024/lab-07/04-cross-encoder-example.ipynb)

Students will learn how to use a cross-encoder model to generate similarity scores between two pieces of text. They
will see how the model can bring substantial artificial intelligence to bear on a simple task and compare performance
to a less specialized model (an embedding model).

## Lab Activity 5

[Your Own Application](05-your-application.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marr75/wecodekc-sientific-computing/blob/main/2024/lab-07/05-your-application.ipynb)

Students will design and implement their own application of a cross-encoder model. They will be encouraged to think
creatively about how to apply this technology to a problem of their choosing.
