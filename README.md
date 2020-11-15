This project is still in progess and will be updated regularly.

# A semi-supervised algorithm to solve Named Entity Recognition problems with limited annotated data
## Presentation

This project adress the problem of Named Entities Recognition (NER) tagging of sentences. The task is to tag each token in a given sentence with an appropriate tag.
Since annotated training data is scarce in this field, we introduce a graph-based semi-supervised algorithm to leverage unannotated data. 
This algorithm extends self-training by estimating the labels of unlabeled data and then using those labels for retraining.

Consider that we want to extract named entities from a text. Our dataset is composed of a large amount of unlabelled data, but a small amount of labelled data. First step is to introduce a CRF model.

## CRF model

We introduce a basic CRF model to categorize our tokens.

## Semi-supervised learning

Our semi-supervised algorithm uses the following steps to estimate the posterior probabilities of the unlabeled data :
    
   * It computes the CRF marginals,
   * Constructs a graph of tokens based on their semantic similarity,
   * Performs a transductive label propagation on the graph,
   * Uses the transductive predictions to interpolate the CRF marginal.

### Graph contruction

Our algorithm first constructs a graph of tokens based on their semantic similarity. The total size of the graph is equal to the number of tokens in both labeled data and unlabeled. The tokens are modeled with word embeddings of 5-grams centered by the current token. This feature vector is calculated by averaging the word vectors multiplied by their TF-IDF scores.

### Label propagation
In order to take advantage of the large quantity of unlabeled data, we perform a transductive **label propagation** over our graph. This choice implies to create a *n x n* matrix which is mostly impossible with a large value of *n*. Remember that the total size of the graph is equal to the total number of tokens. For now, I haven't figured out a way of doing this (yet).

## Interpolate with the CRF marginals

In order to estimate the posterior which can then be used in the CRF training, the commonly used approach is to interpolate the transductive predictions with the CRF marginals using a mixing coefficient.


<img src="https://github.com/samuelgoutin/NLP/blob/master/interpolate.PNG" width="170" height="30">


* p is the posterior probabilities
* b si the transductive predictions
* c is the CRF marginals


# Example

You can find my python implementation of this approach with the [GBC dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) in the [main.ipynb](https://github.com/samuelgoutin/NLP/blob/master/main.ipynb) file.
# Clone the project
```
  $ git clone https://github.com/samuelgoutin/NLP.git
```

# Bibliography 

[Scientific Information Extraction with Semi-supervised Neural Tagging](https://luanyi.github.io/YiLuan_files/keyphrase2017.pdf) from Yi Luan, Mari Ostendorf and Hannaneh Hajishirzi
