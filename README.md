# website-chatbot

## Roadmap

1. Build Question Answering model (always attempts to find answer to question in the context)
2. Build Machine Reading Comprehension model (can choose not to answer a question based on the context)
3. Build a website that can host the model to answer questions about me!

### Deployment Roadmap

1. Build Django backend
2. Create frontend
3. Create Docker interface
4. Buy VPS and point it to domain

### Still To Do

* Finalize MRC real-time pipeline (with accounting for database field lengths, incorporating answerability thresholds, etc.)
* Optimize code structure
* Train model
* Make front-end look better
* Prepare for deployment (Daphne, secret keys, etc.)
* Run the site from my Ubuntu VM
* Set up CI/CD pipeline (not sure if this is done on VM or wait until VPS)
* Get the VPS and run it there

## References

* SQuAD 2.0 - https://arxiv.org/pdf/1806.03822.pdf
* Retrospective Reader - https://arxiv.org/pdf/2001.09694.pdf
* DistilBERT - https://arxiv.org/pdf/1910.01108.pdf
