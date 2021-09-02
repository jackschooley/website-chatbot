import pandas as pd
import json

with open("data/train-v2.0.json") as file:
    train_json = json.load(file)
    
#I think i'll use each topic as its own batch
topics = train_json["data"]

def create_batch_df(topic):
    contexts = []
    questions = []
    answer_texts = []
    answer_starts = []
    
    paragraphs = topic["paragraphs"]
    for paragraph in paragraphs:
        qas = paragraph["qas"]
        for i in range(len(qas)):
            answers = qas[i]["answers"]
            if len(answers) == 1:
                contexts.append(paragraph["context"])
                questions.append(qas[i]["question"])
                answer_texts.append(answers[0]["text"])
                answer_starts.append(answers[0]["answer_start"])
    
    batch_dict = {"contexts": contexts, "questions": questions, 
                  "answer_texts": answer_texts, "answer_starts": answer_starts}
    batch_df = pd.DataFrame(batch_dict)
    return batch_df

all_batch_dfs = [create_batch_df(topic) for topic in topics]
batch_dfs = [df for df in all_batch_dfs if not df.empty]