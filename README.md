# website-chatbot

## About

This is the code for my website that hosts my Machine Reading Comprehension (MRC) model. This is a deep learning model for NLP that not only can answer questions about a given
passage, but it also refrains from answering questions that it does not know the answer to.

* Downloaded SQuAD 2.0 dataset and set up a preprocessing pipeline
* Utilized pretrained DistilBERT transformer for transfer learning
* Reproduced architecture from Retrospective Reader
* Trained model with AdamW
* Built site with Django to host model
* Deployed with Docker

## Roadmap

* General code optimizations and organization
* Prepare for deployment (Daphne, secret keys, etc.)
* Run the site from my Ubuntu VM
* Set up CI/CD pipeline (not sure if this is done on VM or wait until VPS)
* Get the VPS and run it there

## References

* SQuAD 2.0 - https://arxiv.org/pdf/1806.03822.pdf
* DistilBERT - https://arxiv.org/pdf/1910.01108.pdf
* Retrospective Reader - https://arxiv.org/pdf/2001.09694.pdf
* AdamW - https://arxiv.org/pdf/1711.05101.pdf
