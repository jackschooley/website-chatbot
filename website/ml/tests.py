import evaluation
import preprocessing
import torch
import transformers
import unittest

class TestPreprocessing(unittest.TestCase):
    json_file = preprocessing.fetch_json(True)
    df = preprocessing.create_df(json_file)
    tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
    max_length = 512
    
    def test_get_answer_positions(self):
        """Test normal functioning of the get_answer_positions function, where
        the questions are all answerable and don't need truncation"""
        context_starts = [9, 14, 16, 12, 10, 14, 11, 11, 9, 13]
        start_positions = [75, 68, 143, 58, 78, 94, 134, 101, 77, 84]
        end_positions = [78, 70, 143, 60, 79, 97, 136, 102, 78, 85]
        
        df_subset = self.df.iloc[0:10]
        output = preprocessing.tokenize_contexts(df_subset, self.tokenizer, 
                                                 self.max_length)
        
        input_ids = output["input_ids"]
        offsets_mapping = output["offset_mapping"]
        for i in range(input_ids.size(0)):
            token_ids = input_ids[i]
            offset_mapping = offsets_mapping[i]
            answer_text = df_subset.at[i, "answer_text"]
            answer_start = df_subset.at[i, "answer_start"]
            token_positions = preprocessing.get_token_positions(token_ids,
                                                                offset_mapping,
                                                                answer_text,
                                                                answer_start)
            
            context_token, start_token, end_token = token_positions
            self.assertEqual(context_token, context_starts[i])
            
            # start and end tokens are returned as lists
            self.assertEqual(start_token[0], start_positions[i])
            self.assertEqual(end_token[0], end_positions[i])
            
    def test_impossible(self):
        """Test the get_answer_positions function for impossible questions"""
        df_subset = self.df.iloc[2236:2237].reset_index(drop = True)
        output = preprocessing.tokenize_contexts(df_subset, self.tokenizer, 
                                                 self.max_length)
        
        input_ids = output["input_ids"]
        token_ids = input_ids[0]
        context_token = preprocessing.get_token_positions(token_ids)
        self.assertEqual(context_token, 12)
            
    def test_truncation(self):
        """Test the get_answer_positions function when truncation is necessary"""
        df_subset = self.df.iloc[2877:2878].reset_index(drop = True)
        output = preprocessing.tokenize_contexts(df_subset, self.tokenizer, 
                                                 self.max_length)
        
        input_ids = output["input_ids"]
        offsets_mapping = output["offset_mapping"]
        
        token_ids = input_ids[0]
        offset_mapping = offsets_mapping[0]
        answer_text = df_subset.at[0, "answer_text"]
        answer_start = df_subset.at[0, "answer_start"]
        token_positions = preprocessing.get_token_positions(token_ids,
                                                            offset_mapping,
                                                            answer_text,
                                                            answer_start)
        
        self.assertEqual(token_positions, None)
    
class TestEvaluation(unittest.TestCase):
    
    def test_decode_token_logits(self):
        start_logits = torch.tensor([[0.2, -0.8, 0.7, 0.6, 0.1, -0.224],
                                     [0.9, 0.7, 0.53, -0.886, 0.4, 0.8]])
        end_logits = torch.tensor([[0.875, 0.862, -0.1, -0.8, 0.6, 0.4],
                                   [-0.852, 0.3, 0.12, 0.9, 0.6, -0.26]])
        
        context_starts = torch.tensor([2, 3])
        start_positions = torch.tensor([2, 4])
        end_positions = torch.tensor([4, 4])
        start_tokens, end_tokens = evaluation.decode_token_logits(start_logits, 
                                                                  end_logits, 
                                                                  context_starts)
        
        self.assertTrue(torch.equal(start_tokens, start_positions))
        self.assertTrue(torch.equal(end_tokens, end_positions))
    
    def test_set_answerable_threshold(self):
        scores = torch.tensor([0.4, 0.8, 2.1, -0.551, -0.896, -0.7, 0.1038, 0.7])
        is_impossibles = torch.tensor([1, 1, 0, 0, 1, 0, 0, 1])
        best_threshold = evaluation.set_answerable_threshold(scores, is_impossibles, False)
        self.assertAlmostEqual(best_threshold, 0.4)

if __name__ == "__main__":
    unittest.main()