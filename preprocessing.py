import pandas as pd
from batch import BatchIterator
    
def create_df(json_file):
    contexts = []
    questions = []
    answer_texts = []
    answer_starts = []
    
    for topic in json_file["data"]:
        paragraphs = topic["paragraphs"]
        for paragraph in paragraphs:
            qas = paragraph["qas"]
            for i in range(len(qas)):
                answers = qas[i]["answers"]
                if len(answers) >= 1:
                    contexts.append(paragraph["context"])
                    questions.append(qas[i]["question"])
                    answer_texts.append(answers[0]["text"])
                    answer_starts.append(answers[0]["answer_start"])
    
    df = pd.DataFrame({"contexts": contexts, "questions": questions, 
                       "answer_texts": answer_texts, "answer_starts": answer_starts})
    return df

def tokenize_contexts(df, tokenizer):
    contexts = df["contexts"].tolist()
    questions = df["questions"].tolist()
    output = tokenizer(questions, contexts, padding = True, return_tensors = "pt", 
                       return_attention_mask = True, return_offsets_mapping = True)
    return output

def get_token_positions(token_ids, offset_mapping, answer_text, answer_start):
    answer_end = answer_start + len(answer_text)
    
    #skip the question and get to the context
    sep_token_id = 102
    for i, token_id in enumerate(token_ids):
        if token_id.item() == sep_token_id:
            context_start = i + 1
            break
    
    #find the start token in the context
    for j in range(context_start, len(token_ids)):
        token_start, token_end = offset_mapping[j]
        if token_start >= answer_start:
            if token_start > answer_start:
                j -= 1
            start_position = j
            break
        
    #find the end token in the context
    for k in range(start_position, len(token_ids)):
        token_start, token_end = offset_mapping[k]
        if token_end >= answer_end:
            end_position = k
            break
        
    return context_start, start_position, end_position

def preprocess(json_file, tokenizer, max_examples = 10000):
    df = create_df(json_file)
    example_start = 0
    batch_iterator = BatchIterator()
    while example_start < df.shape[0]:
        
        example_end = example_start + max_examples
        print("Processing through example", example_end)
        
        output = tokenize_contexts(df.iloc[example_start:example_end], tokenizer)
        input_ids = output["input_ids"]
        context_starts = []
        start_positions = []
        end_positions = []
        
        for i in range(input_ids.size(0)):
            token_ids = input_ids[i]
            offset_mappings = output["offset_mapping"][i]
            answer_text = df.at[example_start + i, "answer_texts"]
            answer_start = df.at[example_start + i, "answer_starts"]
            token_positions = get_token_positions(token_ids, offset_mappings,
                                                  answer_text, answer_start)
            context_starts.append(token_positions[0])
            start_positions.append(token_positions[1])
            end_positions.append(token_positions[2])
            
        batch_iterator.add_examples(input_ids, output["attention_mask"],
                                     context_starts, start_positions, end_positions)
        example_start = example_end
    return batch_iterator