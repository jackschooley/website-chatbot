import random
import torch

class BatchIterator:
    
    def __init__(self, input_ids, attention_mask, max_length = 512, sep_id = 102, 
                 random_seed = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self._max_length = max_length
        self._sep_id = sep_id
        
        if random_seed:
            self._random_seed = random_seed
        else:
            self._random_seed = None
            
        if self.input_ids.size(1) > self._max_length:
            self.truncate = True
        else:
            self.truncate = False
        
    def _truncate(self):
        split_input_ids = list(self.input_ids.split(1))
        split_attention_mask = list(self.attention_mask.split(1))
        split_context_starts = list(self.context_starts.split(1))
        split_start_positions = list(self.start_positions.split(1))
        split_end_positions = list(self.end_positions.split(1))
        
        #filter out all the examples where the answer is too long
        i = 0
        while i < self.input_ids.size(0):
            try:
                end_position = split_end_positions[i]
            except IndexError:
                break
            if end_position >= self._max_length:
                split_input_ids.pop(i)
                split_attention_mask.pop(i)
                split_context_starts.pop(i)
                split_start_positions.pop(i)
                split_end_positions.pop(i)
                i -= 1
            i += 1
            
        #make sure the [SEP] token is at the end if truncated
        self.input_ids = torch.cat(split_input_ids)[:, :self._max_length]
        end = self._max_length - 1
        for i in range(self.input_ids.size(0)):
            if self.input_ids[i, end] != 0:
                self.input_ids[i, end] = self._sep_id
        
        self.attention_mask = torch.cat(split_attention_mask)[:, :self._max_length]
        self.context_starts = torch.cat(split_context_starts)
        self.start_positions = torch.cat(split_start_positions)
        self.end_positions = torch.cat(split_end_positions)
        
    def _shuffle(self):
        shuffled_ids = []
        shuffled_mask = []
        shuffled_contexts = []
        shuffled_starts = []
        shuffled_ends = []
        
        if self._random_seed is None:
            self._random_seed = 1
        random.seed(self._random_seed)
        
        n = self.input_ids.size(0)
        order = random.sample(range(n), n)
        for i in order:
            shuffled_ids.append(self.input_ids[i, :].unsqueeze(0))
            shuffled_mask.append(self.attention_mask[i, :].unsqueeze(0))
            shuffled_contexts.append(self.context_starts[i].item())
            shuffled_starts.append(self.start_positions[i].item())
            shuffled_ends.append(self.end_positions[i].item())
            
        self.input_ids = torch.cat(shuffled_ids)
        self.attention_mask = torch.cat(shuffled_mask)
        self.context_starts = torch.LongTensor(shuffled_contexts)
        self.start_positions = torch.LongTensor(shuffled_starts)
        self.end_positions = torch.LongTensor(shuffled_ends)
        self._random_seed += 1
        
    def add_positions(self, context_starts, start_positions, end_positions):
        self.context_starts = torch.LongTensor(context_starts)
        self.start_positions = torch.LongTensor(start_positions)
        self.end_positions = torch.LongTensor(end_positions)
        if self.truncate:
            self._truncate()
        
    def get_batches(self, batch_size, train = False):
        if train:
            self._shuffle()
        
        start = 0
        while start <= self.input_ids.size(0):
            end = start + batch_size
            batch_ids = self.input_ids[start:end, :]
            batch_mask = self.attention_mask[start:end, :]
            batch_starts = self.start_positions[start:end]
            batch_ends = self.end_positions[start:end]
            
            if train:
                yield (batch_ids, batch_mask, batch_starts, batch_ends)
            else:
                batch_contexts = self.context_starts[start:end]
                yield (batch_ids, batch_mask, batch_contexts, batch_starts, batch_ends)
                
            start += batch_size