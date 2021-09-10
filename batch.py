import torch

class BatchIterator:
    
    def __init__(self, input_ids, attention_mask, max_length = 512, sep_id = 102):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self._max_length = max_length
        self._sep_id = sep_id
        
    def _truncate(self):
        split_input_ids = list(self.input_ids.split(1))
        split_attention_mask = list(self.attention_mask.split(1))
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
        self.start_positions = torch.cat(split_start_positions)
        self.end_positions = torch.cat(split_end_positions)
        
    def add_positions(self, start_positions, end_positions):
        self.start_positions = torch.LongTensor(start_positions)
        self.end_positions = torch.LongTensor(end_positions)
        self._truncate()
        
    def get_batches(self, batch_size):
        #build in shuffling capability for multiple epochs
        
        start = 0
        while start <= self.input_ids.size(0):
            end = start + batch_size
            batch_ids = self.input_ids[start:end, :]
            batch_mask = self.attention_mask[start:end, :]
            batch_starts = self.start_positions[start:end]
            batch_ends = self.end_positions[start:end]
            yield (batch_ids, batch_mask, batch_starts, batch_ends)
            start += batch_size