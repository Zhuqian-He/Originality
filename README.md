# Measuring Paper Novelty (2024 Publications)

## Overview

Based on the work of Art et al. (2025) and Nicola Melluso's [science-novelty](https://github.com/nicolamelluso/science-novelty), this repository provides code to calculate **novelty indicators for papers published in 2024**. 

The core logic follows the methodology of identifying "new (combinations of) words/phrases" and measuring semantic distance, enabling quantitative evaluation of a paper's innovative contribution.

## Data Requirements

To run the code, you need two types of data:

### 1. OpenAlex Data
- **Purpose**: Obtain basic metadata of papers (e.g., id, title, abstract, publication date) for 2019–2024.
- **How to Get**:
  - Download via [OpenAlex API](https://docs.openalex.org/api/get-started).

### 2. Supplementary Data from Art et al. (2025)
Preprocessed text data to support novelty calculation:
- `papers_words.csv`
- `papers_phrases.csv`
- `new_words.csv`
- `new_phrases.csv`
- `new_word_combs.csv`
- `new_phrase_combs.csv`
- Download link: [Zenodo](https://zenodo.org/records/13902060)

## Methodology Workflow
The novelty calculation is divided into 10 sequential steps:

1. **Screen Papers (2019–2024)**: Filter papers by publication year.

2. **Filter Valid Papers**: Remove papers with missing abstract/title, non-English content, irrelevant fields, and so on.

3. **Text Embedding**: Generate semantic embeddings for paper abstracts/titles.

4. **Semantic Distance Calculation**: Compute distance between the focal paper and all prior papers from the past 5 years.

5. **Text Preprocessing**: Clean text (remove stopwords, punctuation, digits) and extract words/phrases.

6. **New Words**: Identify unique unigrams in 2024 papers that did not appear in prior literature.

7. **New Phrases**: Identify unique noun phrases in 2024 papers that did not appear in prior literature.

8. **New Word Combinations**: Identify unique pairwise combinations of words in 2024 papers that did not appear in prior literature.

9. **New Phrase Combinations**: Identify unique pairwise combinations of phrases in 2024 papers that did not appear in prior literature.

10. **Paper-Level Novelty Metrics**: Aggregate above indicators to get final novelty scores.

### Workflow Diagram
![Calculation Flowchart](https://github.com/Zhuqian-He/Originality/blob/main/Calculation_flowchart.png)  

## Dependencies
Install required packages first, then download spaCy/scispacy models.
- gensim
- joblib
- nltk
- nmslib
- numpy
- pandas
- scispacy
- spacy
- spacy_fastlang
- swifter
- torch
- tqdm
- transformers
- scispacy model: en_core_sci_lg
- spacy model: en_core_web_sm

