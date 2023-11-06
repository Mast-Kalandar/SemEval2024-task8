# %%
from transformers import AutoTokenizer, RobertaModel
import torch
from torch import nn , optim 
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import lightning as L
from pytorch_lightning.loggers import WandbLogger

# %%
tokenizer = AutoTokenizer.from_pretrained("roberta-base" , truncation=True, max_length=512 )
model = RobertaModel.from_pretrained("roberta-base")


# %%
model

# %%

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

# %%
class LSTM_RoBERTA(nn.Module):
    def __init__(self,roberta_path = "roberta-base",lstm_hidden_size=256,lstm_layers=2,lstm_dropout=0.2 , num_classes=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_path)
        self.lstm = nn.LSTM(self.roberta.config.hidden_size,lstm_hidden_size,lstm_layers,batch_first=True,dropout=lstm_dropout,bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_size*2,num_classes)
        self.dropout = nn.Dropout(0.2)
        for param in self.roberta.parameters():
            param.requires_grad = False
        for params in self.roberta.pooler.parameters():
            params.requires_grad = True
        print("Model Loaded")
        print("Total_trainable_params : ",sum(p.numel() for p in self.parameters() if p.requires_grad))
        print("Total_untrainable_params : ",sum(p.numel() for p in self.parameters() if not p.requires_grad))
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids, attention_mask)
        roberta_output = roberta_output.last_hidden_state
        lstm_output, (h_n,c_n) = self.lstm(roberta_output)
        logits = self.linear(lstm_output[:, -1, :])
        # print(logits.shape,"logits")
        return logits

# %%
class DatasetLM(torch.utils.data.Dataset):
    def __init__(self, dataset, tokeniser):
        self.dataset = dataset
        self.tokeniser = tokeniser
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # print(self.tokeniser(self.dataset[index]['text'], return_tensors="pt")['input_ids'])
        return (self.tokeniser(self.dataset[index]['text'], return_tensors="pt", truncation=True, max_length=512), self.dataset[index]['label'])



# %%
train_data_path = "/scratch/jainit/SubtaskA/subtaskA_train_monolingual.jsonl"
test_data_path = "/scratch/jainit/SubtaskA/subtaskA_dev_monolingual.jsonl"
df_train = pd.read_json(train_data_path, lines=True)
df_test = pd.read_json(test_data_path, lines=True)

# %%
train_data = df_train.to_dict('records')
test_data = df_test.to_dict('records')

# %%
train_dataset = DatasetLM(train_data, tokenizer)
test_dataset = DatasetLM(test_data, tokenizer)

# %%


# %%
def collate_fn(batch):
    input_ids = [x[0]['input_ids'].squeeze(0) for x in batch]
    # print(input_ids)
    attention_mask = [x[0]['attention_mask'].squeeze(0) for x in batch]
    labels = [x[1] for x in batch]
    # print(batch)
    # input_ids = torch.stack(input_ids)
    # attention_mask = torch.stack(attention_mask)
    # print(input_ids.shape)
    # print([x.shape for x in input_ids])
    inputs_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    # print(inputs_ids.shape)
    x = {'input_ids': inputs_ids, 'attention_mask': attention_mask}
    y = torch.Tensor(labels)
    return (x, y)

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# %%
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS


class LitModel(L.LightningModule):
    def __init__(self,train_loader , val_loader , model_path, lr=2e-5, hidden_size=256, lstm_layers=2, lstm_dropout=0.2, num_classes=2):
        super().__init__()
        self.model_path = model_path
        self.lr = lr
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.num_classes = num_classes
        self.model = LSTM_RoBERTA(model_path, hidden_size, lstm_layers, lstm_dropout, num_classes)
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask):
        # print(input_ids.shape , attention_mask.shape , "Here are the shapes")
        out = self.model(input_ids, attention_mask)
        # print(out.shape , "Here is the output$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(x, y )
        # print( y.shape, "Here are wasd############" )
        logits = self.forward(x['input_ids'], x['attention_mask'])
        # print(logits.shape , "Here are the logits")
        loss = F.cross_entropy(logits, y.long())
        # print(logits.shape , y.shape)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True   )
        
        return loss
         
        # print("##############")
        # return None
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x['input_ids'], x['attention_mask'])
        loss = F.cross_entropy(logits, y.long())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True   )
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x['input_ids'], x['attention_mask'])
        loss = F.cross_entropy(logits, y.long())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True   )
        return loss
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.val_loader
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr , weight_decay=0.01 , eps=1e-6)
        

# %%
trainer = L.Trainer(max_epochs=10, devices=1, num_nodes=1, logger=WandbLogger(project="roberta-lstm", log_model=False, name="Full-RUN" ),default_root_dir="robert-lstm")

# %%
model = LitModel.load_from_checkpoint("/scratch/jainit/LSTM-classifier/best_bert_freeze.ckpt" , train_loader=train_loader , val_loader=test_loader , model_path="roberta-base" , lr=2e-5, hidden_size=256, lstm_layers=2, lstm_dropout=0.2, num_classes=2)
# trainer.fit(LitModel(train_loader, test_loader, "roberta-base"))
# trainer.save_checkpoint("last_man.ckpt")
# %%
trainer.test(model)

# %%


# %%


# %%



