import torch
import transformers
from .ml.evaluation import decode_token_logits, get_answer
from .ml.model import MRCModel
from .ml.preprocessing import get_token_positions

def mrc_pipeline(question, context, model, delta = 0.5):
    tokenizer = transformers.DistilBertTokenizerFast("mrc/ml/vocab.txt")
    sigmoid = torch.nn.Sigmoid()
    model.eval()
    
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
    
    ids = input_ids[0] #need to make this a one dimensional tensor
    context_start = get_token_positions(ids, offset_mapping)
    start_token, end_token = decode_token_logits(start_logits, end_logits, [context_start])
    
    if bool_probs.item() < delta:
        answer = get_answer(ids, tokenizer, start_token, end_token)
    else:
        answer = "Don't know my guy."
    return answer

if __name__ == "__main__":
    question = "Are you a reply guy?"
    context = "Actually I'm not a reply guy."
    
    distilbert_config = transformers.DistilBertConfig(n_layers = 3, n_heads = 6,
                                                      dim = 384, hidden_dim = 1536)
    model = MRCModel(distilbert_config)
    model.load_state_dict(torch.load("ml/model_weights.pth"))
    
    answer = mrc_pipeline(question, context, model)
    print(answer)