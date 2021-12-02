import random
import torch

class BatchIterator:
    
    def __init__(self, train, max_length = 512, sep_id = 102, pad_id = 0, 
                 random_seed = None):
        
        self._train = train
        self._max_length = max_length
        self._sep_id = sep_id
        self._pad_id = pad_id
        
        if random_seed:
            self._random_seed = random_seed
        else:
            self._random_seed = None
            
        self.input_ids = None
        self.attention_mask = None
        self.context_starts = None
        self.start_positions = None
        self.end_positions = None
        self.is_impossibles = None
        
    def _shuffle(self):
        
        if self._random_seed is None:
            self._random_seed = 1
        random.seed(self._random_seed)
        
        # create the permutation
        n = self.input_ids.size(0)
        order = random.sample(range(n), n)
        
        # apply the permutation
        self.input_ids = self.input_ids[order, :]
        self.attention_mask = self.attention_mask[order, :]
        self.context_starts = self.context_starts[order]
        self.start_positions = self.start_positions[order]
        self.end_positions = self.end_positions[order]
        self.is_impossibles = self.is_impossibles[order]
        
        # increment the seed for the next epoch
        self._random_seed += 1
        
    def add_examples(self, input_ids, attention_mask, context_starts, 
                     start_positions, end_positions, is_impossibles):
        
        contexts = torch.LongTensor(context_starts)
        impossibles = torch.LongTensor(is_impossibles)
        if self._train:
            starts = torch.LongTensor(start_positions)
            ends = torch.LongTensor(end_positions)
        
        if self.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, input_ids])
            self.attention_mask = torch.cat([self.attention_mask, attention_mask])
            self.context_starts = torch.cat([self.context_starts, contexts])
            self.is_impossibles = torch.cat([self.is_impossibles, impossibles])
            if self._train:
                self.start_positions = torch.cat([self.start_positions, starts])
                self.end_positions = torch.cat([self.end_positions, ends])
            else:
                self.start_positions.extend(start_positions)
                self.end_positions.extend(end_positions)
        else:
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.context_starts = contexts
            self.is_impossibles = impossibles
            if self._train:
                self.start_positions = starts
                self.end_positions = ends
            else:
                self.start_positions = start_positions
                self.end_positions = end_positions
        
    def get_batches(self, batch_size):
        if self._train:
            self._shuffle()
        
        start = 0
        while start <= self.input_ids.size(0):
            end = start + batch_size
            batch_ids = self.input_ids[start:end, :]
            batch_mask = self.attention_mask[start:end, :]
            batch_contexts = self.context_starts[start:end]
            
            if self._train:
                batch_starts = self.start_positions[start:end]
                batch_ends = self.end_positions[start:end]
                batch_impossibles = self.is_impossibles[start:end]
                yield (batch_ids, 
                       batch_mask,
                       batch_contexts,
                       batch_starts, 
                       batch_ends, 
                       batch_impossibles)
            else:
                yield (batch_ids, batch_mask, batch_contexts)
                
            start = end