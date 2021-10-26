import torch
import transformers

context = "Actually I'm not a reply guy."

#tokenize the question
def create_input(question, context, tokenizer):
    output = tokenizer(question, context, padding = True, return_tensors = "pt",
                       return_attention_mask = True, return_offsets_mapping = True)
    return output
    
model = torch.load("model.pth")
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")

def mrc_pipeline(input_ids, attention_mask):
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
        
    start_logits = model_output.start_logits
    end_logits = model_output.end_logits
    bool_logits = model_output.bool_logits
    return start_logits, end_logits, bool_logits