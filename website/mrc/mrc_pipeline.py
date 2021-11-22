import numpy as np
import random
import torch

def get_context_start(token_ids, sep_token_id):
    for i, token_id in enumerate(token_ids):
        if token_id.item() == sep_token_id:
            context_start = i + 1
            break
    return context_start

def decode_token_logits(start_logits, end_logits, context_start):
    sequence_length = start_logits.size(1)
    adjusted_length = sequence_length - context_start
    matrix_dims = (adjusted_length, adjusted_length)
    
    # create a matrix to represent joint logits
    starts = start_logits[:, context_start:].squeeze(0).unsqueeze(1)
    ends = end_logits[:, context_start:]
    matrix = starts + ends
    
    # create upper triangular matrix to ensure start <= end and take argmax
    triu_matrix = np.triu(matrix.numpy())
    triu_matrix[triu_matrix == 0] = np.NINF
    flat_argmax = np.argmax(triu_matrix)
    raw_start, raw_end = np.unravel_index(flat_argmax, matrix_dims)
    
    # correct for shifted index
    start_token = raw_start + context_start
    end_token = raw_end + context_start
    
    return start_token, end_token

def get_answer(token_ids, tokenizer, start_token, end_token):
    ids = token_ids[start_token:end_token + 1].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    string = tokenizer.convert_tokens_to_string(tokens)
    return string

def generate_unanswerable_response():
    possible_responses = [
        "Good question.",
        "No idea honestly.",
        "Machines are too dumb to answer that question."
    ]
    response = random.choice(possible_responses)
    return response

def mrc_pipeline(question, context, tokenizer, model, delta):
    output = tokenizer(question, context, 
                       padding = "max_length", 
                       max_length = model.sequence_length,
                       return_tensors = "pt",
                       return_attention_mask = True)
    
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    
    token_ids = input_ids.squeeze(0)
    context_start = get_context_start(token_ids, tokenizer.sep_token_id)
    context_starts = torch.tensor([context_start])
    
    model.eval()
    with torch.no_grad():
        model_output = model(input_ids, attention_mask, context_starts)
    
    scores = model_output.scores
    start_logits = model_output.start_logits
    end_logits = model_output.end_logits
    start_token, end_token = decode_token_logits(start_logits, end_logits, context_start)
    
    if scores.item() < delta:
        answer = get_answer(token_ids, tokenizer, start_token, end_token)
    else:
        answer = generate_unanswerable_response()
    return answer