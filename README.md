# website-chatbot

This is the source code for https://jackschooley.co.

## About

The site hosts a Machine Reading Comprehension (MRC) model which does question answering on a few different topics about me. Most
crucially, the model refrains from answering questions that it does not know the answer to. I tackled this NLP task using deep learning
and deployed my model on a self-hosted website.

So what exactly went into this project?

* Downloaded SQuAD 2.0 dataset and set up a preprocessing pipeline
* Utilized pretrained DistilBERT transformer for transfer learning
* Reproduced architecture from Retrospective Reader paper
* Trained model to achieve 56% EM and 57% F1 on the development set
* Built site with Django to host model
* Designed site format and layout with HTML and CSS
* Deployed with Daphne as a Docker container
* Hosted site on a VPS with a Traefik reverse proxy

## Directory Structure

The `ml` folder has all of the code involved for the machine learning development process. There you'll find the code used for
training, developing, and evaluating the model as well as the saved weights and hyperparameters.

The `mrc` folder contains the code for the questions portion of the site, including the real-time evaluation pipeline for the MRC task.
You can find the text that the model uses to answer questions in the `topics/contexts` subfolder.

## References

* SQuAD 2.0 - https://arxiv.org/pdf/1806.03822.pdf
* DistilBERT - https://arxiv.org/pdf/1910.01108.pdf
* Retrospective Reader - https://arxiv.org/pdf/2001.09694.pdf
* AdamW - https://arxiv.org/pdf/1711.05101.pdf