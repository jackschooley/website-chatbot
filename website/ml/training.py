import preprocessing
import torch
import transformers
from model import MRCModel

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
configuration = transformers.DistilBertConfig()
weights = torch.load("distilbert_pretrained_weights.pth")

# learning rate of 0.0005 is too low for SGD
learning_rate = 0.0003
batch_size = 4
epochs = 2

batch_iterator = preprocessing.preprocess(tokenizer)
model = MRCModel(configuration, weights).cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

model.train()
for epoch in range(epochs):
    print("Epoch", epoch)
    batches = batch_iterator.get_batches(batch_size)
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
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print("Batch", i, "loss is", loss.item())
        torch.cuda.empty_cache()
    
    # save weights after every epoch just to be safe
    torch.save(model.state_dict(), "model_weights" + str(epoch) + ".pth")