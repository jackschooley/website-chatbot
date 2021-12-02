import preprocessing
import torch
import torch.optim as optim
import transformers
from model import MRCModel

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
batch_iterator = preprocessing.preprocess(tokenizer)

configuration = transformers.DistilBertConfig()
weights = torch.load("distilbert_pretrained_weights.pth")
model = MRCModel(configuration, weights).cuda()

# warmup for first epoch with learning rate 1/10th the size of inital rate
learning_rate = 0.00002
optimizer = optim.AdamW(model.parameters(), learning_rate)
lr_lambda = lambda epoch: 10 ** (epoch - 1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# gpu can only hold 2 examples at a time
gpu_batch_size = 2
batch_size = 32
gpu_batch_cycles = int(batch_size / gpu_batch_size)
epochs = 2

model.train()
for epoch in range(epochs):
    print("Epoch", epoch)
    batches = batch_iterator.get_batches(gpu_batch_size)
    batch_cycle = 0
    batch_loss = 0
    
    for i, batch in enumerate(batches):
        input_ids = batch[0].cuda()
        attention_mask = batch[1].cuda()
        context_starts = batch[2].cuda()
        start_positions = batch[3].cuda()
        end_positions = batch[4].cuda()
        is_impossibles = batch[5].cuda()
        
        model_output = model(input_ids, attention_mask, context_starts,
                             start_positions, end_positions, is_impossibles)
        loss = model_output.loss
        
        loss.backward()
        batch_loss += loss.item()
        
        #accumulate gradients until target batch size is reached
        if (i + 1) % gpu_batch_cycles == 0:
            optimizer.step()
            optimizer.zero_grad(True)
        
            if batch_cycle % 50 == 0:
                print("Batch", batch_cycle, "loss is", batch_loss / gpu_batch_cycles)
            
            batch_cycle += 1
            batch_loss = 0
    scheduler.step()
    
# save trained weights
torch.save(model.state_dict(), "model_weights.pth")