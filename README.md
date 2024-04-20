# nlp_api
We propose an API that queries task-specific fine-tuned BERT models for 1) emotion recognition, 2) named entity recognition, and 3) sentence similarity and provide inference example notebooks for each task.

## Usage

We recommend cloning this repository from Google Colab on a GPU runtime.

## PLEASE READ
These notebooks are runnable but depend on a model which is too large to upload to Github. If you wish to perform inference, you must run the corresponding task notebook before running an inference notebook. For instance, first open and run all cells in **`emotion.ipynb`**. Following this, it is possible to open and run the cells in **`emotion_inference.ipynb`**. If running on Colab, be sure to mount your Google Drive and clone the repository from your root directory.

## Required libraries

If running locally, be sure to install the following Python libraries (using `pip install` or `conda install`) before beginning:

- SpeechRecognition
- moviepy
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

Inspiration for the sentence similarity and named entity recognition (NER) tasks was taken from corresponding tutorials from <a href="https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt">Chapter 3</a> and <a href="https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt">Chapter 7</a> of the Hugging Face Transformers Course.