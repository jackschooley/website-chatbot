import json
import pandas as pd
from batch import BatchIterator

def fetch_json(train, folder = "data/"):
    if train:
        filename = folder + "train-v2.0.json"
    else:
        filename = folder + "dev-v2.0.json"
    
    with open(filename) as file:
        json_file = json.load(file)
    return json_file
    
def create_df(json_file, ignore_index = -999):
    contexts = []
    questions = []
    answer_texts = []
    answer_starts = []
    is_impossibles = []
    
    for topic in json_file["data"]:
        paragraphs = topic["paragraphs"]
        for paragraph in paragraphs:
            qas = paragraph["qas"]
            for qa in qas:
                answers = qa["answers"]
                contexts.append(paragraph["context"])
                questions.append(qa["question"])
                
                # for the train json there will only be one answer at most
                # for the dev json there are 3-5 answers that are not unique
                if len(answers) >= 1:
                    texts = []
                    starts = []
                    for answer in answers:
                        text = answer["text"]
                        if text not in texts:
                            texts.append(text)
                            starts.append(answer["answer_start"])
                    answer_texts.append(tuple(texts))
                    answer_starts.append(tuple(starts))
                    is_impossibles.append(0)
                
                # the question is marked as impossible, with no answer
                else:
                    answer_texts.append("")
                    answer_starts.append(ignore_index)
                    is_impossibles.append(1)
    
    df = pd.DataFrame({
        "context": contexts, 
        "question": questions, 
        "answer_text": answer_texts, 
        "answer_start": answer_starts,
        "is_impossible": is_impossibles
    })
    
    return df

def tokenize_contexts(df, tokenizer, max_length):
    contexts = df["context"].tolist()
    questions = df["question"].tolist()
    output = tokenizer(questions, contexts, 
                       padding = "max_length",
                       truncation = "only_second", 
                       max_length = max_length,
                       return_tensors = "pt", 
                       return_attention_mask = True, 
                       return_offsets_mapping = True)
    return output

def get_answer_positions(token_ids, offset_mapping, context_start, answer_text, 
                         answer_start, answer_end):
    
    start_position = None
    end_position = None
    
    # find the start token in the context
    for j in range(context_start, len(token_ids)):
        token_start, token_end = offset_mapping[j]
        if token_start >= answer_start:
            if token_start > answer_start:
                j -= 1
            start_position = j
            break
    
    # if the answer got truncated then the start position won't be found
    if not start_position:
        return start_position, end_position
        
    # find the end token in the context
    for k in range(start_position, len(token_ids)):
        token_start, token_end = offset_mapping[k]
        if token_end >= answer_end:
            end_position = k
            break
    
    return start_position, end_position

def get_token_positions(token_ids, offset_mapping, answer_text = None, 
                        answer_start = None, sep_token_id = 102):
    
    # skip the question and get to the context
    for i, token_id in enumerate(token_ids):
        if token_id.item() == sep_token_id:
            context_start = i + 1
            break
    
    # if the question is impossible, only the context start is needed
    if not answer_text:
        return context_start
    
    # the answer text and start are given in tuples to account for multiples
    starts_position = []
    ends_position = []
    for i in range(len(answer_text)):
        text = answer_text[i]
        start = answer_start[i]
        end = start + len(text)
        start_position, end_position = get_answer_positions(token_ids, 
                                                            offset_mapping,
                                                            context_start,
                                                            text, start, end)
        if end_position:
            starts_position.append(start_position)
            ends_position.append(end_position)
        else:
            # the start and/or end position got truncated
            return
        
    return context_start, starts_position, ends_position

def preprocess(tokenizer, train = True, max_examples = 10000, max_length = 512,
               ignore_index = -999):
    
    json_file = fetch_json(train)
    df = create_df(json_file)
    batch_iterator = BatchIterator(train)
    
    example_start = 0
    while example_start < df.shape[0]:
        example_end = example_start + max_examples
        print("Processing through example", example_end)
        
        # running the entire dataset at once uses too much memory
        df_subset = df.iloc[example_start:example_end]
        output = tokenize_contexts(df_subset, tokenizer, max_length)
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        
        context_starts = []
        start_positions = []
        end_positions = []
        is_impossibles = df_subset["is_impossible"].reset_index(drop = True)
        
        # need to keep track of which observations are kept
        observations_kept = []
        for i in range(input_ids.size(0)):
            token_ids = input_ids[i]
            if is_impossibles[i] == 0:
                answer_text = df.at[example_start + i, "answer_text"]
                answer_start = df.at[example_start + i, "answer_start"]
                offsets_mapping = output["offset_mapping"][i]
                token_positions = get_token_positions(token_ids, 
                                                      offsets_mapping,
                                                      answer_text, 
                                                      answer_start)
                
                if token_positions:
                    observations_kept.append(i)
                    context_starts.append(token_positions[0])
                    starts_position = token_positions[1]
                    ends_position = token_positions[2]
                else:
                    # this observation got truncated, so will be thrown out
                    continue
                
                if train:
                    # destroy the list and just take the only value
                    start_positions.append(starts_position[0])
                    end_positions.append(ends_position[0])
                else:
                    # keep the tuple and its potentially multiple values
                    start_positions.append(tuple(starts_position))
                    end_positions.append(tuple(ends_position))
            else:
                observations_kept.append(i)
                context_start = get_token_positions(token_ids, offsets_mapping)
                context_starts.append(context_start)
                start_positions.append(ignore_index)
                end_positions.append(ignore_index)
        
        impossibles = is_impossibles.iloc[observations_kept].tolist()
        batch_iterator.add_examples(input_ids[observations_kept, :], 
                                    attention_mask[observations_kept, :],
                                    context_starts, start_positions, 
                                    end_positions, impossibles)
        
        example_start = example_end
        
    return batch_iterator