import json
import numpy as np
import preprocessing
import torch
import transformers
from model import MRCModel
    
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

def set_answerable_threshold(bool_probs, impossibles):
    n = len(impossibles)
    question_sets = zip(bool_probs, impossibles)
    question_sets_sorted = sorted(question_sets, key = lambda x: x[0])
    best_threshold = 0
    best_accuracy = sum(impossibles).item() / n
    for question_set in question_sets_sorted:
        threshold = question_set[0].item()
        decision = bool_probs >= threshold
        accuracy = sum(decision == impossibles).item() / n
        if accuracy > best_accuracy:
            best_threshold = threshold
    return best_threshold

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

def evaluate(model, tokenizer, batch_iterator, batch_size):
    start_logits = None
    end_logits = None
    bool_logits = None
    for batch in batch_iterator.get_batches(batch_size):
        input_ids = batch[0].cuda()
        attention_mask = batch[1].cuda()
        start_positions = batch[2]
        end_positions = batch[3]
        
        with torch.no_grad():
            model_output = model(input_ids, attention_mask)
            
        batch_start_logits = model_output.start_logits.cpu()
        batch_end_logits = model_output.end_logits.cpu()
        batch_bool_logits = model_output.bool_logits.cpu()
        
        if start_logits is not None:
            start_logits = torch.cat([start_logits, batch_start_logits], 0)
            end_logits = torch.cat([end_logits, batch_end_logits], 0)
            bool_logits = torch.cat([bool_logits, batch_bool_logits], 0)
        else:
            start_logits = batch_start_logits
            end_logits = batch_end_logits
            bool_logits = batch_bool_logits
            
    sigmoid = torch.nn.Sigmoid()
    bool_probs = sigmoid(bool_logits)
    
    context_starts = batch_iterator.context_starts
    start_tokens, end_tokens = decode_token_logits(start_logits, end_logits, 
                                                   context_starts)
    
    impossibles = batch_iterator.impossibles
    delta = set_answerable_threshold(bool_probs, impossibles)
    
    predicted_answers = []
    true_answers = []
    for i in range(input_ids.size(0)):
        if bool_probs[i] >= delta:
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
    
    accuracy = exact_matches(predicted_answers, true_answers)
    return accuracy, delta

if __name__ == "__main__":
    with open("data/dev-v2.0.json") as file:
        dev_json = json.load(file)
    
    distilbert_config = transformers.DistilBertConfig(n_layers = 3, n_heads = 6, 
                                                      dim = 384, hidden_dim = 1536)
    model = MRCModel(distilbert_config).cuda()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    
    tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
    batch_iterator = preprocessing.preprocess(dev_json, tokenizer)
    batch_size = 16
    
    accuracy, delta = evaluate(model, tokenizer, batch_iterator, batch_size)
    print("Exact matches:", accuracy)
    print("Answerability threshold used:", delta)