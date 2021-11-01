import torch
import transformers
from evaluation import decode_token_logits, get_answer
from preprocessing import get_token_positions

question = "Are you a reply guy?"
context = "Actually I'm not a reply guy."
    
model = torch.load("model.pth").cpu() #this should be fixed at some point
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
sigmoid = torch.nn.Sigmoid()
delta = 0.5

def mrc_pipeline(question, context, tokenizer):
    output = tokenizer(question, context, padding = True, return_tensors = "pt",
                       return_attention_mask = True, return_offsets_mapping = True)
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    offset_mapping = output["offset_mapping"]
    
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
        
    start_logits = model_output.start_logits
    end_logits = model_output.end_logits
    bool_logits = model_output.bool_logits
    bool_probs = sigmoid(bool_logits)
    
    context_start = list(get_token_positions(input_ids[0], offset_mapping))
    start_token, end_token = decode_token_logits(start_logits, end_logits, context_start)
    
    if bool_probs.item() >= delta:
        answer = get_answer(input_ids, tokenizer, start_token, end_token)
    else:
        answer = "Don't know my guy."
    return answer