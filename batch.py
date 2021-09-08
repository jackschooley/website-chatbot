import torch

class Batch:
    
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
    def add_positions(self, start_positions, end_positions):
        self.start_positions = torch.LongTensor(start_positions)
        self.end_positions = torch.LongTensor(end_positions)