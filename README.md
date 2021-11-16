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

1. Finalize MRC model
2. Write initialization for Django topics (topics, contexts)
3. Write tests for ML development pipeline
4. Finalize MRC real-time pipeline (with accounting for database field lengths, incorporating answerability thresholds, etc.)
5. Optimize code structure
6. Train model
7. Make front-end look better
8. Prepare for deployment (Daphne, secret keys, etc.)
9. Run the site from my Ubuntu VM
10. Set up CI/CD pipeline (not sure if this is done on VM or wait until VPS)
11. Get the VPS and run it there

## References

* SQuAD 2.0 - https://arxiv.org/pdf/1806.03822.pdf
* Retrospective Reader - https://arxiv.org/pdf/2001.09694.pdf
* DistilBERT - https://arxiv.org/pdf/1910.01108.pdf
