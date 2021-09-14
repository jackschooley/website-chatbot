import torch.nn as nn
import transformers

class ModelOutput:
    
    def __init__(self, start_logits, end_logits, loss):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.loss = loss

class MRCModel(nn.Module):
    
    def __init__(self, distilbert_config):
        
        super(MRCModel, self).__init__()
        self.distilbert = transformers.DistilBertModel(distilbert_config)
        self.dropout = nn.Dropout(distilbert_config.qa_dropout)
        self.qa_outputs = nn.Linear(distilbert_config.dim, 2)
        
    def forward(self, input_ids, attention_mask, start_positions = None, 
                end_positions = None):
        
        distilbert_output = self.distilbert(input_ids, attention_mask)
        hidden_states = distilbert_output.last_hidden_state
        dropout_states = self.dropout(hidden_states)
        logits = self.qa_outputs(dropout_states)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_function = nn.CrossEntropyLoss()
            start_loss = loss_function(start_logits, start_positions)
            end_loss = loss_function(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            
        return ModelOutput(start_logits, end_logits, total_loss)