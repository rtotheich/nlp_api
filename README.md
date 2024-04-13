# nlp_api
We propose an API that queries task-specific fine-tuned BERT models for 1) emotion recognition, 2) named entity recognition, and 3) sentence similarity and provide inference example notebooks for each task.

## Usage

We recommend cloning this repository from Google Colab on a GPU runtime.

## Required libraries

If running locally, be sure to install the following Python libraries (using `pip install` or `conda install`) before beginning:

- numpy
- matplotlib
- seaborn
- transformers
- scikitlearn
- umap-learn
- accelerate -U
- datasets
- scipy
- tqdm
- sentence_transformers
- torch

## Attribution

The emotion notebook is largely inspired by chapter 3 of the book <a href="https://www.oreilly.com/library/view/natural-language-processing/9781098136789/">Natural Language Processing with Transformers</a> by Lewis Tunstall, Leandro von Werra, and Thomas Wolf. The book is available through O'Reilly Media.
