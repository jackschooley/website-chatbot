import torch
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
        self.nonlinearity = nn.Tanh()
        self.dim = distilbert_config.dim
        self.sequence_length = distilbert_config.max_position_embeddings
        
        self.start_linear = nn.Linear(self.dim, 1)
        self.end_linear = nn.Linear(self.dim + 1, 1)
        self.sketchy = nn.Linear(self.dim, 1)
        self.deep = nn.Linear(self.sequence_length * 2, 1)
        self.answerable = nn.Linear(2, 1)
        
    def forward(self, input_ids, attention_mask, start_positions = None, 
                end_positions = None, impossibles = None):
        
        distilbert_output = self.distilbert(input_ids, attention_mask)
        hidden_states = distilbert_output.last_hidden_state #(bs, sl, hs)
        dropout_states = self.dropout(hidden_states) #(bs, sl, hs)
        
        start_logits = self.start_linear(dropout_states) #(bs, sl, 1)
        end_linear_inputs = torch.cat([dropout_states, start_logits], 2) #(bs, sl, hs + 1)
        end_logits = self.end_linear(end_linear_inputs) #(bs, sl, 1)
        
        cls_hidden_state = hidden_states[:, 0, :] # (bs, 1, hs)
        sketchy_logits = self.sketchy(cls_hidden_state) #(bs, 1)
        sketchy = self.nonlinearity(sketchy_logits) #(bs, 1)
        
        start_logits_squeezed = start_logits.squeeze(2) #(bs, sl)
        end_logits_squeezed = end_logits.squeeze(2) #(bs, sl)
        answer_logits = torch.cat([start_logits_squeezed, end_logits_squeezed], 1) #(bs, 2sl)
        answer = self.nonlinearity(answer_logits) #(bs, 2sl)
        
        #this is necessary during real-time evaluation only
        padding_n = self.sequence_length * 2 - answer.size(1)
        padded_answer = nn.functional.pad(answer, (0, padding_n))
        
        deep_logits = self.deep(padded_answer) #(bs, 1)
        deep = self.nonlinearity(deep_logits) #(bs, 1)
        
        answerable_logits = torch.cat([sketchy, deep], 1) #(bs, 2)
        bool_logits = self.answerable(answerable_logits).squeeze(1) #(bs, )
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            
            qa_loss_function = nn.CrossEntropyLoss(ignore_index = -999)
            start_loss = qa_loss_function(start_logits_squeezed, start_positions)
            end_loss = qa_loss_function(end_logits_squeezed, end_positions)
            
            bool_loss_function = nn.BCEWithLogitsLoss()
            possibility_loss = bool_loss_function(bool_logits, impossibles)
            
            total_loss = (start_loss + end_loss + possibility_loss) / 3
            
        return ModelOutput(start_logits_squeezed, end_logits_squeezed, bool_logits, 
                           total_loss)