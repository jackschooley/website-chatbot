import random
import pandas as pd
import torch

class BatchIterator:
    
    def __init__(self, train, max_length = 512, sep_id = 102, 
                 pad_id = 0, random_seed = None):
        
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
        """This method will only be called when training"""
        shuffled_ids = []
        shuffled_mask = []
        shuffled_contexts = []
        shuffled_starts = []
        shuffled_ends = []
        shuffled_impossis = []
        
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
            shuffled_impossis.append(self.is_impossibles[i].item())
            
        self.input_ids = torch.cat(shuffled_ids)
        self.attention_mask = torch.cat(shuffled_mask)
        self.context_starts = torch.LongTensor(shuffled_contexts)
        self.start_positions = torch.LongTensor(shuffled_starts)
        self.end_positions = torch.LongTensor(shuffled_ends)
        self.is_impossibles = torch.FloatTensor(shuffled_impossis)
        self._random_seed += 1
        
    def add_examples(self, input_ids, attention_mask, context_starts, 
                     start_positions, end_positions, is_impossibles):
        
        def filtered(ends_position):
            """This function accounts for the fact that the ends_position
            column can have an int or a tuple"""
            if type(ends_position) == tuple:
                return max(ends_position) < self._max_length
            return ends_position < self._max_length
        
        # filter out the observations that got their answers truncated
        df = pd.DataFrame({
            "context_start": context_starts,
            "starts_position": start_positions,
            "ends_position": end_positions,
            "is_impossibles": is_impossibles
        })
        
        filtered_obs = df["ends_position"].apply(filtered)
        filtered_ids = input_ids[filtered_obs, :]
        filtered_mask = attention_mask[filtered_obs, :]
        filtered_df = df.loc[filtered_obs]
        
        filtered_contexts = filtered_df["context_start"].tolist()
        filtered_starts = filtered_df["starts_position"].tolist()
        filtered_ends = filtered_df["ends_position"].tolist()
        filtered_impossibles = filtered_df["is_impossibles"].tolist()
        
        contexts = torch.LongTensor(filtered_contexts)
        if self._train:
            starts = torch.LongTensor(filtered_starts)
            ends = torch.LongTensor(filtered_ends)
            impossibles = torch.LongTensor(filtered_impossibles)
        
        if self.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, filtered_ids])
            self.attention_mask = torch.cat([self.attention_mask, filtered_mask])
            self.context_starts = torch.cat([self.context_starts, contexts])
            if self._train:
                self.start_positions = torch.cat([self.start_positions, starts])
                self.end_positions = torch.cat([self.end_positions, ends])
                self.is_impossibles = torch.cat([self.is_impossibles, impossibles])
            else:
                self.start_positions.extend(filtered_starts)
                self.end_positions.extend(filtered_ends)
                self.is_impossibles.extend(filtered_impossibles)
        else:
            self.input_ids = filtered_ids
            self.attention_mask = filtered_mask
            self.context_starts = contexts
            if self._train:
                self.start_positions = starts
                self.end_positions = ends
                self.is_impossibles = impossibles
            else:
                self.start_positions = filtered_starts
                self.end_positions = filtered_ends
                self.is_impossibles = filtered_impossibles
        
    def get_batches(self, batch_size):
        if self._train:
            self._shuffle()
        
        start = 0
        while start <= self.input_ids.size(0):
            end = start + batch_size
            batch_ids = self.input_ids[start:end, :]
            batch_mask = self.attention_mask[start:end, :]
            
            if self._train:
                batch_starts = self.start_positions[start:end]
                batch_ends = self.end_positions[start:end]
                batch_impossis = self.is_impossibles[start:end]
                yield (batch_ids, batch_mask, batch_starts, batch_ends, batch_impossis)
            else:
                yield (batch_ids, batch_mask)
                
            start = end