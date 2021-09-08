import json
import pandas as pd
import torch
import transformers
from batch import Batch

with open("data/train-v2.0.json") as file:
    train_json = json.load(file)
    
#I think i'll use each topic as its own batch
topics = train_json["data"]

def tokenize_contexts(df, tokenizer):
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
        if token_start >= answer_start:
            if token_start > answer_start:
                i -= 1
            start_position = i
            break
    for j in range(start_position, len(context_ids)):
        token_start, token_end = offset_mapping[j]
        if token_end >= answer_end:
            end_position = j
            break
    return start_position, end_position

def preprocess_one(df, tokenizer):
    output = tokenize_contexts(df, tokenizer)
    input_ids = output["input_ids"]
    batch = Batch(input_ids, output["attention_mask"])
    start_positions = []
    end_positions = []
    for i in range(input_ids.size(0)):
        context_ids = input_ids[i]
        offset_mappings = output["offset_mapping"][i]
        answer_text = df.at[i, "answer_texts"]
        answer_start = df.at[i, "answer_starts"]
        start_position, end_position = get_token_positions(context_ids, offset_mappings,
                                                           answer_text, answer_start)
        start_positions.append(start_position)
        end_positions.append(end_position)
    batch.add_positions(start_positions, end_positions)
    return batch
    
def create_batch(topic, tokenizer):
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
    if not batch_df.empty:
        batch = preprocess_one(batch_df, tokenizer)
        return batch

def preprocess_all(topics, tokenizer):
    batches = []
    for topic in topics:
        batch = create_batch(topic, tokenizer)
        if batch is not None:
            batches.append(batch)
    return batches

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
configuration = transformers.DistilBertConfig(max_position_embeddings = 1024)
learning_rate = 0.0001

batches = preprocess_all(topics, tokenizer)
model = transformers.DistilBertForQuestionAnswering(configuration).cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

model.train()
max_batch_size = 1
for i, batch in enumerate(batches):
    input_ids = batch.input_ids[:max_batch_size, :].cuda()
    attention_mask = batch.attention_mask[:max_batch_size, :].cuda()
    start_positions = batch.start_positions[:max_batch_size].cuda()
    end_positions = batch.end_positions[:max_batch_size].cuda()
    
    model_output = model(input_ids, attention_mask, start_positions = start_positions,
                         end_positions = end_positions)
    loss = model_output.loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Batch", i, "loss is", loss.item())
    torch.cuda.empty_cache()