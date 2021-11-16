import preprocessing
import torch
import transformers
from model import MRCModel

tokenizer = transformers.DistilBertTokenizerFast("vocab.txt")
configuration = transformers.DistilBertConfig(n_layers = 3, n_heads = 6,
                                              dim = 384, hidden_dim = 1536)

learning_rate = 0.0001
batch_size = 16
epochs = 1

batch_iterator = preprocessing.preprocess(tokenizer)
model = MRCModel(configuration).cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

model.train()
for epoch in range(epochs):
    print("Epoch", epoch)
    batches = batch_iterator.get_batches(batch_size)
    for i, batch in enumerate(batches):
        input_ids = batch[0].cuda()
        attention_mask = batch[1].cuda()
        start_positions = batch[2].cuda()
        end_positions = batch[3].cuda()
        impossibles = batch[4].cuda()
        
        model_output = model(input_ids, attention_mask, start_positions,
                             end_positions, impossibles)
        loss = model_output.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("Batch", i, "loss is", loss.item())

torch.save(model.state_dict(), "model_weights.pth")