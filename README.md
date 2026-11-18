# Lyrics NLP Project

NLP project for lyrics preprocessing, text analysis, similarity search, and genre classification using Python and machine learning.

## Overview

This project explores natural language processing techniques applied to English song lyrics.

The project covers the complete workflow from dataset preprocessing and tokenization to TF-IDF vectorization, text similarity search, dimensionality reduction, and machine learning-based genre classification.

The project was developed in Google Colab using Python and several NLP and machine learning libraries.

## Objectives

The main objectives of the project were:

- Clean and preprocess a large collection of song lyrics
- Tokenize and prepare text for machine learning
- Transform lyrics into numerical representations using TF-IDF
- Visualize high-dimensional text representations using PCA
- Build an interactive lyrics similarity search
- Compare different machine learning approaches for binary genre classification
- Evaluate model performance using classification reports and confusion matrices

## Dataset

The project uses the **Genius Lyrics** dataset available through Hugging Face:

`brunokreiner/genius-lyrics`

The dataset contains English song lyrics together with genre and artist-related information.

For the classification task, the original genre information was transformed into a binary target:

- `1` — the song is associated with the `pop` genre
- `0` — the song is not associated with the `pop` genre

The data was divided into:

- 60% training set
- 20% validation set
- 20% test set

## NLP Pipeline

The project follows the following text processing pipeline:

```text
Raw Lyrics
    ↓
Data Cleaning
    ↓
Text Normalization
    ↓
Stopword Removal
    ↓
Tokenization
    ↓
Train / Validation / Test Split
    ↓
TF-IDF Vectorization
    ↓
├── PCA Visualization
├── Similarity Search
└── Machine Learning Classification
