import json
import numpy as np
import preprocessing
import torch
import transformers

with open("data/dev-v2.0.json") as file:
    dev_json = json.load(file)
    
def decode_token_logits(start_logits, end_logits, context_starts):
    batch_size, sequence_length = start_logits.size()
    start_tokens = torch.zeros(batch_size, dtype = torch.long)
    end_tokens = torch.zeros(batch_size, dtype = torch.long)
    
    for i in range(batch_size):
        context_start = context_starts[i]
        adjusted_length = sequence_length - context_start
        
        #create a matrix to represent joint logits
        start_logits_unsqueezed = start_logits[i, context_start:].unsqueeze(1)
        end_logits_unsqueezed = end_logits[i, context_start:].unsqueeze(0)
        matrix = start_logits_unsqueezed + end_logits_unsqueezed
        
        #create upper triangular matrix to ensure start <= end and take argmax
        triu_matrix = np.triu(matrix.numpy())
        triu_matrix[triu_matrix == 0] = np.NINF
        flat_argmax = np.argmax(triu_matrix)
        raw_start, raw_end = np.unravel_index(flat_argmax, 
                                              (adjusted_length, adjusted_length))
        
        #correct for shifted index
        start_tokens[i] = raw_start + context_start
        end_tokens[i] = raw_end + context_start
    return start_tokens, end_tokens

def get_answer(input_ids, tokenizer, start, end):
    ids = input_ids[start:end + 1].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    string = tokenizer.convert_tokens_to_string(tokens)
    return string

def exact_matches(predicted_answers, true_answers):
    correct = 0
    for i in range(len(predicted_answers)):
        if predicted_answers[i] == true_answers[i]:
            correct += 1
    return correct
    
model = torch.load("model.pth").cuda()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
batch_iterator = preprocessing.preprocess(dev_json, tokenizer)
sigmoid = torch.nn.Sigmoid()

batch_size = 16
delta = 0.5

predicted_answers = []
true_answers = []
for batch in batch_iterator.get_batches(batch_size):
    input_ids = batch[0].cuda()
    attention_mask = batch[1].cuda()
    context_starts = batch[2]
    start_positions = batch[3]
    end_positions = batch[4]
    impossibles = batch[5]
    
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
        
    start_logits = model_output.start_logits.cpu()
    end_logits = model_output.end_logits.cpu()
    
    bool_logits = model_output.bool_logits.cpu()
    bool_probs = sigmoid(bool_logits)
    
    start_tokens, end_tokens = decode_token_logits(start_logits, end_logits, context_starts)
    
    for i in range(input_ids.size(0)):
        if bool_logits[i] >= delta:
            predicted_answer = get_answer(input_ids[i, :], tokenizer, start_tokens[i], 
                                          end_tokens[i])
            predicted_answers.append(predicted_answer)
        else:
            predicted_answers.append("")
        
        if impossibles[i] == 0:
            true_answer = get_answer(input_ids[i, :], tokenizer, start_positions[i], 
                                     end_positions[i])
            true_answers.append(true_answer)
        else:
            true_answers.append("")
        
print("Exact matches:", exact_matches(predicted_answers, true_answers))