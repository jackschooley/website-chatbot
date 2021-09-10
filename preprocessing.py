import json
import pandas as pd
import torch
import transformers
from batch import BatchIterator

with open("data/train-v2.0.json") as file:
    train_json = json.load(file)
    
def create_df(json_file):
    contexts = []
    questions = []
    answer_texts = []
    answer_starts = []
    
    for topic in json_file["data"]:
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
    
    df = pd.DataFrame({"contexts": contexts, "questions": questions, 
                       "answer_texts": answer_texts, "answer_starts": answer_starts})
    return df

def tokenize_contexts(df, tokenizer):
    contexts = df["contexts"].tolist()
    questions = df["questions"].tolist()
    output = tokenizer(questions, contexts, padding = True, return_tensors = "pt", 
                       return_attention_mask = True, return_offsets_mapping = True)
    return output

def get_token_positions(token_ids, offset_mapping, answer_text, answer_start):
    answer_end = answer_start + len(answer_text)
    
    #skip the question and get to the context
    sep_token_id = 102
    for i, token_id in enumerate(token_ids):
        if token_id.item() == sep_token_id:
            context_start = i + 1
            break
    
    #find the start token in the context
    for j in range(context_start, len(token_ids)):
        token_start, token_end = offset_mapping[j]
        if token_start >= answer_start:
            if token_start > answer_start:
                j -= 1
            start_position = j
            break
        
    #find the end token in the context
    for k in range(start_position, len(token_ids)):
        token_start, token_end = offset_mapping[k]
        if token_end >= answer_end:
            end_position = k
            break
        
    return start_position, end_position

def preprocess(json_file, tokenizerm, max_examples):
    df = create_df(json_file)
    #this is a quick fix because the entire df is too big
    output = tokenize_contexts(df.iloc[:max_examples], tokenizer)
    input_ids = output["input_ids"]
    batch_iterator = BatchIterator(input_ids, output["attention_mask"])
    start_positions = []
    end_positions = []
    for i in range(input_ids.size(0)):
        token_ids = input_ids[i]
        offset_mappings = output["offset_mapping"][i]
        answer_text = df.at[i, "answer_texts"]
        answer_start = df.at[i, "answer_starts"]
        start_position, end_position = get_token_positions(token_ids, offset_mappings,
                                                           answer_text, answer_start)
        start_positions.append(start_position)
        end_positions.append(end_position)
    batch_iterator.add_positions(start_positions, end_positions)
    return batch_iterator

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
configuration = transformers.DistilBertConfig(n_layers = 3, n_heads = 6,
                                              dim = 384, hidden_dim = 1536)

learning_rate = 0.0001
batch_size = 16
max_examples = 40000

batch_iterator = preprocess(train_json, tokenizer, max_examples)
model = transformers.DistilBertForQuestionAnswering(configuration).cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

model.train()
for i, batch in enumerate(batch_iterator.get_batches(batch_size)):
    input_ids = batch[0].cuda()
    attention_mask = batch[1].cuda()
    start_positions = batch[2].cuda()
    end_positions = batch[3].cuda()
    
    model_output = model(input_ids, attention_mask, start_positions = start_positions,
                         end_positions = end_positions)
    loss = model_output.loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Batch", i, "loss is", loss.item())