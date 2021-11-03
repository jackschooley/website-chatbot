import random
import torch

class BatchIterator:
    
    def __init__(self, max_length = 512, sep_id = 102, pad_id = 0, random_seed = None):
        
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
        self.impossibles = None
        
    def _pad(self, input_ids, attention_mask):
        length, width = input_ids.size()
        missing_width = self._max_length - width
        pad_tensor = torch.full((length, missing_width - 1), self._pad_id)
        id_end_tensor = torch.full((length, 1), self._sep_id)
        mask_end_tensor = torch.zeros_like(id_end_tensor, dtype = torch.long)
        ids = torch.cat([input_ids, pad_tensor, id_end_tensor], 1)
        mask = torch.cat([attention_mask, pad_tensor, mask_end_tensor], 1)
        return ids, mask
        
    def _shuffle(self):
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
            shuffled_impossis.append(self.impossibles[i].item())
            
        self.input_ids = torch.cat(shuffled_ids)
        self.attention_mask = torch.cat(shuffled_mask)
        self.context_starts = torch.LongTensor(shuffled_contexts)
        self.start_positions = torch.LongTensor(shuffled_starts)
        self.end_positions = torch.LongTensor(shuffled_ends)
        self.impossibles = torch.FloatTensor(shuffled_impossis)
        self._random_seed += 1
        
    def _truncate(self, input_ids, attention_mask, context_starts,
                  start_positions, end_positions, impossibles):
        
        split_input_ids = list(input_ids.split(1))
        split_attention_mask = list(attention_mask.split(1))
        split_context_starts = list(context_starts.split(1))
        split_start_positions = list(start_positions.split(1))
        split_end_positions = list(end_positions.split(1))
        split_impossibles = list(impossibles.split(1))
        
        #filter out all the examples where the answer is too long
        i = 0
        while i < input_ids.size(0):
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
                split_impossibles.pop(i)
                i -= 1
            i += 1
            
        #make sure the [SEP] token is at the end if truncated
        ids = torch.cat(split_input_ids)[:, :self._max_length]
        end = self._max_length - 1
        for i in range(ids.size(0)):
            if ids[i, end] != 0:
                ids[i, end] = self._sep_id
        
        mask = torch.cat(split_attention_mask)[:, :self._max_length]
        contexts = torch.cat(split_context_starts)
        starts = torch.cat(split_start_positions)
        ends = torch.cat(split_end_positions)
        impossis = torch.cat(split_impossibles)
        
        truncated_inputs = (ids, mask, contexts, starts, ends, impossis)
        return truncated_inputs
        
    def add_examples(self, input_ids, attention_mask, context_starts, 
                     start_positions, end_positions, impossibles):
        
        contexts = torch.LongTensor(context_starts)
        starts = torch.LongTensor(start_positions)
        ends = torch.LongTensor(end_positions)
        impossis = torch.FloatTensor(impossibles)
        
        sequence_length = input_ids.size(1)
        if sequence_length > self._max_length:
            truncated_inputs = self._truncate(input_ids, attention_mask, contexts, 
                                              starts, ends, impossis)
            ids = truncated_inputs[0]
            mask = truncated_inputs[1]
            contexts = truncated_inputs[2]
            starts = truncated_inputs[3]
            ends = truncated_inputs[4]
            impossis = truncated_inputs[5]
        elif sequence_length < self._max_length:
            ids, mask = self._pad(input_ids, attention_mask)
        else:
            ids = input_ids
            mask = attention_mask
        
        if self.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, ids])
            self.attention_mask = torch.cat([self.attention_mask, mask])
            self.context_starts = torch.cat([self.context_starts, contexts])
            self.start_positions = torch.cat([self.start_positions, starts])
            self.end_positions = torch.cat([self.end_positions, ends])
            self.impossibles = torch.cat([self.impossibles, impossis])
        else:
            self.input_ids = ids
            self.attention_mask = mask
            self.context_starts = contexts
            self.start_positions = starts
            self.end_positions = ends
            self.impossibles = impossis
        
    def get_batches(self, batch_size, train = False):
        if train:
            self._shuffle()
        
        start = 0
        while start <= self.input_ids.size(0):
            end = start + batch_size
            batch_ids = self.input_ids[start:end, :]
            batch_mask = self.attention_mask[start:end, :]
            
            if train:
                batch_starts = self.start_positions[start:end]
                batch_ends = self.end_positions[start:end]
                batch_impossis = self.impossibles[start:end]
                yield (batch_ids, batch_mask, batch_starts, batch_ends, batch_impossis)
            else:
                yield (batch_ids, batch_mask)
                
            start = end