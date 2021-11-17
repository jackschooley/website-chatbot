import preprocessing
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
            
class TestModel(unittest.TestCase):
    
    def test_dimensions(self):
        """This test goes through the model architecture and makes sure all
        the tensor dimensions behave like they're supposed to"""
        pass
            
class TestEvaluation(unittest.TestCase):
    
    def test_decode_token_logits(self):
        pass
    
    def test_get_answerable_threshold(self):
        pass

if __name__ == "__main__":
    unittest.main()