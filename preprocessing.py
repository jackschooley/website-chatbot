import json
import pandas as pd
import transformers

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

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")

def tokenize_contexts(df):
    contexts = df["contexts"].tolist()
    questions = df["questions"].tolist()
    #right now i have contexts before questions, might change
    output = tokenizer(contexts, questions, padding = True, return_tensors = "pt", 
                       return_attention_mask = True, return_offsets_mapping = True)
    return output

def get_token_positions(context_ids, offset_mapping, answer_text, answer_start):
    answer_end = answer_start + len(answer_text)
    for i in range(len(context_ids)):
        token_start, token_end = offset_mapping[i]
        if answer_start >= token_start:
            if answer_start > token_start:
                i -= 1
            start_position = i
            break
    for j in range(start_position, len(context_ids)):
        token_start, token_end = offset_mapping[i]
        if answer_end >= token_end:
            end_position = j
            break
    return start_position, end_position

def preprocess(df):
    output = tokenize_contexts(df)
    start_positions = []
    end_positions = []
    for i in range(output["input_ids"].size(0)):
        context_ids = output["input_ids"][i]
        offset_mappings = output["offset_mapping"][i]
        answer_text = df.at[i, "answer_texts"]
        answer_start = df.at[i, "answer_starts"]
        start_position, end_position = get_token_positions(context_ids, offset_mappings,
                                                           answer_text, answer_start)
        start_positions.append(start_position)
        end_positions.append(end_position)
    df["start_tokens"] = start_positions
    df["end_tokens"] = end_positions
        
configuration = transformers.DistilBertConfig()
model = transformers.DistilBertForQuestionAnswering(configuration)