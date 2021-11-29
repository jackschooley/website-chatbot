import numpy as np
import torch
import torch.nn as nn
import transformers

class ModelOutput:
    
    def __init__(self, scores, start_logits = None, end_logits = None, loss = None):
        self.scores = scores
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.loss = loss
        
class Sketchy(nn.Module):
    
    def __init__(self, dim):
        super(Sketchy, self).__init__()
        self.linear = nn.Linear(dim, 2)
        
    def forward(self, cls_hidden_state, is_impossibles = None):
        sketchy_logits = self.linear(cls_hidden_state)
        score_ext = sketchy_logits[:, 1] - sketchy_logits[:, 0]
        
        sketchy_loss = None
        if is_impossibles is not None:
            sketchy_loss_fct = nn.CrossEntropyLoss()
            sketchy_loss = sketchy_loss_fct(sketchy_logits, is_impossibles)
        return ModelOutput(score_ext, loss = sketchy_loss)
    
class Intensive(nn.Module):
    
    def __init__(self, dim, ignore_index):
        super(Intensive, self).__init__()
        self.ignore_index = ignore_index
        
        self.qa = nn.Linear(dim, 2)
        self.linear = nn.Linear(dim, 1)
        
    def _get_score_diff(self, context_starts, start_logits, end_logits):
        score_has = torch.zeros_like(context_starts, dtype = torch.float)
        for i in range(context_starts.size(0)):
            context_start = context_starts[i]
            
            # create a matrix by broadcasting
            start_logits_unsqueezed = start_logits[i, context_start:].unsqueeze(1)
            end_logits_unsqueezed = end_logits[i, context_start:].unsqueeze(0)
            matrix = start_logits_unsqueezed + end_logits_unsqueezed
            
            # create upper triangular matrix to ensure start <= end
            triu_matrix = np.triu(matrix.cpu().detach().numpy())
            triu_matrix[triu_matrix == 0] = np.NINF
            score_has[i] = torch.tensor(np.amax(triu_matrix))
        
        score_null = start_logits[:, 0] + end_logits[:, 0]
        score_diff = score_null - score_has
        return score_diff
        
    def forward(self, dropout_states, cls_hidden_state, context_starts, 
                start_positions, end_positions, is_impossibles):
        
        qa_logits = self.qa(dropout_states)
        start_logits = qa_logits[:, :, 0]
        end_logits = qa_logits[:, :, 1]
        
        intensive_logits = self.linear(cls_hidden_state)
        score_diff = self._get_score_diff(context_starts, start_logits, end_logits)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            qa_loss_fct = nn.CrossEntropyLoss(ignore_index = self.ignore_index)
            start_loss = qa_loss_fct(start_logits, start_positions)
            end_loss = qa_loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            
            intensive_loss_fct = nn.BCEWithLogitsLoss()
            intensive_loss = intensive_loss_fct(intensive_logits.squeeze(1), 
                                                is_impossibles.float())
            
            # weight loss appropriately
            total_loss = 0.5 * qa_loss + 0.5 * intensive_loss
        return ModelOutput(score_diff, start_logits, end_logits, total_loss)

class MRCModel(nn.Module):
    
    def __init__(self, distilbert_config, weights = None, ignore_index = -999):
        
        super(MRCModel, self).__init__()
        self.distilbert = transformers.DistilBertModel(distilbert_config)
        self.dim = distilbert_config.dim
        self.sequence_length = distilbert_config.max_position_embeddings
        
        # load pretrained distilbert weights during training
        if weights is not None:
            self.distilbert.load_state_dict(weights())
        
        self.sketchy = Sketchy(self.dim)
        self.dropout = nn.Dropout(distilbert_config.qa_dropout)
        self.intensive = Intensive(self.dim, ignore_index)
        
    def forward(self, input_ids, attention_mask, context_starts,
                start_positions = None, end_positions = None, 
                is_impossibles = None):
        
        # run inputs through distilbert and run sketchy module
        distilbert_output = self.distilbert(input_ids, attention_mask)
        hidden_states = distilbert_output.last_hidden_state
        cls_hidden_state = hidden_states[:, 0, :]
        sketchy_output = self.sketchy(cls_hidden_state, is_impossibles)
        sketchy_scores = sketchy_output.scores
        
        # run dropout layer for qa and run intensive module
        dropout_states = self.dropout(hidden_states)
        intensive_output = self.intensive(dropout_states, cls_hidden_state, 
                                          context_starts, start_positions, 
                                          end_positions, is_impossibles)
        
        intensive_scores = intensive_output.scores
        start_logits = intensive_output.start_logits
        end_logits = intensive_output.end_logits
        
        # weight sketchy and intensive modules appropriately
        scores = 0.5 * sketchy_scores + 0.5 * intensive_scores
        
        loss = None
        if sketchy_output.loss is not None and intensive_output.loss is not None:
            loss = 0.5 * sketchy_output.loss + 0.5 * intensive_output.loss
        return ModelOutput(scores, start_logits, end_logits, loss)