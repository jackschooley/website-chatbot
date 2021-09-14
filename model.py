import torch.nn as nn
import transformers

class ModelOutput:
    
    def __init__(self, start_logits, end_logits, bool_logits, loss):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.bool_logits = bool_logits
        self.loss = loss

class MRCModel(nn.Module):
    
    def __init__(self, distilbert_config):
        
        super(MRCModel, self).__init__()
        self.distilbert = transformers.DistilBertModel(distilbert_config)
        self.dropout = nn.Dropout(distilbert_config.qa_dropout)
        self.qa_outputs = nn.Linear(distilbert_config.dim, 2)
        self.possibility = nn.Linear(distilbert_config.dim, 1)
        
    def forward(self, input_ids, attention_mask, start_positions = None, 
                end_positions = None, impossibles = None):
        
        distilbert_output = self.distilbert(input_ids, attention_mask)
        hidden_states = distilbert_output.last_hidden_state
        dropout_states = self.dropout(hidden_states)
        
        qa_logits = self.qa_outputs(dropout_states)
        start_logits = qa_logits[:, :, 0]
        end_logits = qa_logits[:, :, 1]
        
        cls_hidden_state = hidden_states[:, 0, :]
        bool_logits = self.possibility(cls_hidden_state).squeeze(1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            
            qa_loss_function = nn.CrossEntropyLoss(ignore_index = -999)
            start_loss = qa_loss_function(start_logits, start_positions)
            end_loss = qa_loss_function(end_logits, end_positions)
            
            bool_loss_function = nn.BCEWithLogitsLoss()
            possibility_loss = bool_loss_function(bool_logits, impossibles)
            
            total_loss = (start_loss + end_loss + possibility_loss) / 3
            
        return ModelOutput(start_logits, end_logits, bool_logits, total_loss)