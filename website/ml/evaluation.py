import numpy as np
import pickle
import preprocessing
import torch
import transformers
from model import MRCModel
    
def decode_token_logits(start_logits, end_logits, context_starts):
    examples, sequence_length = start_logits.size()
    start_tokens = torch.zeros(examples, dtype = torch.long)
    end_tokens = torch.zeros(examples, dtype = torch.long)
    
    for i in range(examples):
        context_start = context_starts[i]
        adjusted_length = sequence_length - context_start
        matrix_dims = (adjusted_length, adjusted_length)
        
        # create a matrix to represent joint logits
        start_logits_unsqueezed = start_logits[i, context_start:].unsqueeze(1)
        end_logits_unsqueezed = end_logits[i, context_start:].unsqueeze(0)
        matrix = start_logits_unsqueezed + end_logits_unsqueezed
        
        # create upper triangular matrix to ensure start <= end and take argmax
        triu_matrix = np.triu(matrix.numpy())
        triu_matrix[triu_matrix == 0] = np.NINF
        flat_argmax = np.argmax(triu_matrix)
        raw_start, raw_end = np.unravel_index(flat_argmax, matrix_dims)
        
        # correct for shifted index
        start_tokens[i] = raw_start + context_start
        end_tokens[i] = raw_end + context_start
        
    return start_tokens, end_tokens

def set_answerable_threshold(scores, is_impossibles, output_pickle = True):
    n = len(is_impossibles)
    question_sets = zip(scores, is_impossibles)
    question_sets_sorted = sorted(question_sets, key = lambda x: x[0])
    
    best_threshold = 0
    best_accuracy = sum(is_impossibles).item() / n
    for question_set in question_sets_sorted:
        threshold = question_set[0].item()
        
        # decision = 1 is labeled impossible
        decision = scores >= threshold
        accuracy = sum(decision == is_impossibles).item() / n
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy
    
    # serialize the threshold created
    if output_pickle:
        with open("delta.pickle", "wb") as file:
            pickle.dump(best_threshold, file, pickle.HIGHEST_PROTOCOL)
            
    return best_threshold

def get_answer(input_ids, tokenizer, start, end):
    ids = input_ids[start:end + 1].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    string = tokenizer.convert_tokens_to_string(tokens)
    return string

def get_answers(input_ids, tokenizer, starts, ends):
    answers = []
    for i in range(len(starts)):
        answer = get_answer(input_ids, tokenizer, starts[i], ends[i])
        answers.append(answer)
    return answers

def compute_em(predicted_answers, true_answers):
    n = len(predicted_answers)
    correct = 0
    for i in range(n):
        if predicted_answers[i] in true_answers[i]:
            correct += 1
    em = correct / n
    return em

def compute_max_f1(start_token, end_token, start_positions, end_positions):
    max_f1 = 0
    predicted_tokens = np.array([i for i in range(start_token, end_token + 1)])
    
    # iterate over all the potential correct answers
    for start_position, end_position in zip(start_positions, end_positions):
        tokens = np.array([i for i in range(start_position, end_position + 1)])
        common = np.intersect1d(tokens, predicted_tokens, True)
        correct = common.size
        
        # avoid dividing by 0
        if correct:
            precision = correct / predicted_tokens.size
            recall = correct / tokens.size
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(f1, max_f1)
    return max_f1

def compute_f1(predicted_answers, start_tokens, end_tokens, start_positions, 
               end_positions, is_impossibles):
    f1 = 0
    for i, predicted_answer in enumerate(predicted_answers):
        if is_impossibles[i]:
            if predicted_answer:
                f1 += 1
        else:
            f1 += compute_max_f1(start_tokens[i], end_tokens[i], 
                                 start_positions[i], end_positions[i])
    f1 /= len(predicted_answers)
    return f1

def evaluation_loop(model, batch_iterator, batch_size):
    scores = None
    start_logits = None
    end_logits = None
    for batch in batch_iterator.get_batches(batch_size):
        input_ids = batch[0].cuda()
        attention_mask = batch[1].cuda()
        context_starts = batch[2].cuda()
        
        with torch.no_grad():
            model_output = model(input_ids, attention_mask, context_starts)
            
        batch_start_logits = model_output.start_logits.cpu()
        batch_end_logits = model_output.end_logits.cpu()
        batch_scores = model_output.scores.cpu()
        
        if start_logits is not None:
            scores = torch.cat([scores, batch_scores], 0)
            start_logits = torch.cat([start_logits, batch_start_logits], 0)
            end_logits = torch.cat([end_logits, batch_end_logits], 0)
        else:
            scores = batch_scores
            start_logits = batch_start_logits
            end_logits = batch_end_logits
        
    return scores, start_logits, end_logits

def evaluate(model, tokenizer, batch_iterator, batch_size):
    input_ids = batch_iterator.input_ids
    context_starts = batch_iterator.context_starts
    start_positions = batch_iterator.start_positions
    end_positions = batch_iterator.end_positions
    is_impossibles = batch_iterator.is_impossibles
    
    print("Starting batch evaluation")
    scores, start_logits, end_logits = evaluation_loop(model, batch_iterator, 
                                                       batch_size)
    
    print("Getting start and end tokens")
    start_tokens, end_tokens = decode_token_logits(start_logits, end_logits, 
                                                   context_starts)
    
    print("Setting answerability threshold")
    delta = set_answerable_threshold(scores, is_impossibles)
    
    print("Getting answers")
    predicted_answers = []
    true_answers = []
    for i in range(scores.size(0)):
        if scores[i] < delta:
            predicted_answer = get_answer(input_ids[i, :], 
                                          tokenizer, 
                                          start_tokens[i], 
                                          end_tokens[i])
            predicted_answers.append(predicted_answer)
        else:
            predicted_answers.append("")
        
        if is_impossibles[i] == 0:
            true_answer = get_answers(input_ids[i, :], 
                                      tokenizer, 
                                      start_positions[i], 
                                      end_positions[i])
            true_answers.append(true_answer)
        else:
            true_answers.append([""])
    
    em = compute_em(predicted_answers, true_answers)
    f1 = compute_f1(predicted_answers, start_tokens, end_tokens, 
                    start_positions, end_positions, is_impossibles)
    
    return em, f1, delta, predicted_answers, true_answers

if __name__ == "__main__":
    distilbert_config = transformers.DistilBertConfig()
    model = MRCModel(distilbert_config).cuda()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    
    tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
    batch_iterator = preprocessing.preprocess(tokenizer, False)
    batch_size = 2
    
    output = evaluate(model, tokenizer, batch_iterator, batch_size)
    em, f1, delta, preds, truth = output
    print("EM:", em)
    print("F1:", f1)
    print("Answerability threshold used:", delta)