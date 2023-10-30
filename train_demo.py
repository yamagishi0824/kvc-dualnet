import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_curve
import time
import matplotlib.pyplot as plt

scenario = 'desktop' #'mobile'
sequence_length = 100 #50
EXP = "xxx"
EPOCHS = 1000

batches_per_epoch_train = 16
batches_per_epoch_val = 4
batch_size_train = 2048 #1024
batch_size_val = 2048

def contrastive_loss(x1, x2, label, margin: float = 1.):
    dist = nn.functional.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss

# Classifier Model
class TypeClassifier(nn.Module):
    def __init__(self):
        super(TypeClassifier, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=64)
        
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)

        # Max Pooling
        self.maxpool = nn.AdaptiveMaxPool1d(1)  # Reduce sequence length to 1
        
        self.fc = nn.Linear(64 * 64, 64)

        self.logits = nn.Sequential(
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.maxpool(x)
        x = x.squeeze(-1)
        x = self.logits(x)
        
        return x

# Embedding Model
class Type1DCNNTransformer(nn.Module):
    def __init__(self, sequence_length=sequence_length, nhead=8, num_transformer_layers=1):
        super(Type1DCNNTransformer, self).__init__()
        
        # 1D Conv layers
        self.batch_norm = nn.BatchNorm1d(num_features=sequence_length)
        
        self.conv1 = nn.Conv1d(7, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        
        # Max Pooling
        self.maxpool = nn.AdaptiveMaxPool1d(1)  # Reduce sequence length to 1
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Final linear layers
        self.logits = nn.Sequential(
            nn.Linear(128, 64), 
        )
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = x.transpose(1, 2) # Ensure the sequence length is the last dimension
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Transformer expects input size as (S, N, E) where S is source sequence length, N is batch size, E is feature number.
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = F.relu(x)
        x = x.permute(1, 2, 0) # Revert back to (N, E, S)
        
        x = self.maxpool(x)
        x = x.squeeze(-1)  # Remove the sequence length dimension
        
        x = self.logits(x)
        x = F.softmax(x, dim=1)
        
        return x

class PrepareData:
    def __init__(self, dataset, sequence_length, samples_considered_per_epoch):
        self.data = dataset
        self.len = samples_considered_per_epoch
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        user_idx = random.choice(list(self.data.keys()))
        session_idx = random.choice(list(self.data[user_idx].keys()))
        session_1 = self.data[user_idx][session_idx]
        # session_1 = np.concatenate((session_1, np.zeros((self.sequence_length + 1, np.shape(session_1)[1]))))[:self.sequence_length + 1]
        # print(session_1.shape)
        session_1 = session_1 % 10000
        session_1 = session_1.astype(np.float32)
        session_1 = cv2.resize(session_1, (3, self.sequence_length + 1))
        hl_1 = np.reshape((session_1[:, 1] - session_1[:, 0]) , (np.shape(session_1)[0], 1))
        hl_1_diff = np.reshape(np.diff(session_1[:, 1] - session_1[:, 0]), (self.sequence_length, 1))
        hl_1 = hl_1[:self.sequence_length]
        pl_1 = np.reshape(np.diff(session_1[:, 0]) , (self.sequence_length, 1))
        rl_1 = np.reshape(np.diff(session_1[:, 1]) , (self.sequence_length, 1))
        il_1 = pl_1 - hl_1
        ascii_1 = np.reshape(session_1[:, 2] / 256, (np.shape(session_1)[0], 1))
        ascii_1_diff = np.reshape(np.diff(session_1[:, 2] / 256), (self.sequence_length, 1))
        ascii_1 = ascii_1[:self.sequence_length]
        hl_1, hl_1_diff, pl_1, rl_1, il_1 = hl_1 / 1e3, hl_1_diff / 1e3, pl_1 / 1e3, rl_1 / 1e3, il_1 / 1e3
        session_1_processed = np.concatenate((hl_1, hl_1_diff, pl_1, rl_1, il_1, ascii_1, ascii_1_diff), axis=1)

        label = random.choice([0, 1])
        if label == 0:
            session_idx_2 = random.choice([x for x in list(self.data[user_idx].keys()) if x != session_idx])
            session_2 = self.data[user_idx][session_idx_2]
        else:
            user_idx_2 = random.choice([x for x in list(self.data.keys()) if x != user_idx])
            session_idx_2 = random.choice(list(self.data[user_idx_2].keys()))
            session_2 = self.data[user_idx_2][session_idx_2]
        # session_2 = np.concatenate((session_2, np.zeros((self.sequence_length + 1, np.shape(session_2)[1]))))[:self.sequence_length + 1]
        session_2 = session_2 % 10000
        session_2 = session_2.astype(np.float32)
        session_2 = cv2.resize(session_2, (3, self.sequence_length + 1))
        hl_2 = np.reshape((session_2[:, 1] - session_2[:, 0]) , (np.shape(session_2)[0], 1))
        hl_2_diff = np.reshape(np.diff(session_2[:, 1] - session_2[:, 0]), (self.sequence_length, 1))
        hl_2 = hl_2[:self.sequence_length]
        pl_2 = np.reshape(np.diff(session_2[:, 0]) , (self.sequence_length, 1))
        rl_2 = np.reshape(np.diff(session_2[:, 1]) , (self.sequence_length, 1))
        il_2 = pl_2 - hl_2
        ascii_2 = np.reshape(session_2[:, 2] / 256, (np.shape(session_2)[0], 1))
        ascii_2_diff = np.reshape(np.diff(session_2[:, 2] / 256), (self.sequence_length, 1))
        ascii_2 = ascii_2[:self.sequence_length]
        hl_2, hl_2_diff, pl_2, rl_2, il_2 = hl_2 / 1e3, hl_2_diff / 1e3, pl_2 / 1e3, rl_2 / 1e3, il_2 / 1e3
        session_2_processed = np.concatenate((hl_2, hl_2_diff, pl_2, rl_2, il_2, ascii_2, ascii_2_diff), axis=1)

        return (session_1_processed, session_2_processed), label

    def __len__(self):
        return self.len



# file_loc = '{}/{}_dev_set.npy'.format(scenario, scenario)
file_loc = '../data/KVC_data_for_participants/{}/{}_dev_set.npy'.format(scenario, scenario)
data = np.load(file_loc, allow_pickle=True).item()

dev_set_users = list(data.keys())
random.shuffle(dev_set_users)
train_val_division = 0.8
train_users = dev_set_users[:int(len(dev_set_users)*train_val_division)]
val_users = dev_set_users[int(len(dev_set_users)*train_val_division):]


train_data = copy.deepcopy(data)
for user in list(data.keys()):
    if user not in train_users:
        del train_data[user]
val_data = copy.deepcopy(data)
for user in list(data.keys()):
    if user not in val_users:
        del val_data[user]
del data

try:
    print(len(train_data))
except:
    print(type(train_data))



ds_t = PrepareData(train_data, sequence_length=sequence_length, samples_considered_per_epoch=batches_per_epoch_train*batch_size_train)
ds_v = PrepareData(val_data, sequence_length=sequence_length, samples_considered_per_epoch=batches_per_epoch_val*batch_size_val)

input_data, labels = ds_t[0]
print(input_data[0].shape, input_data[1].shape, labels)
print(input_data[0])
print(len(ds_t))

train_dataloader = DataLoader(ds_t, batch_size=batch_size_train)
val_dataloader = DataLoader(ds_v, batch_size=batch_size_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# for embedding model
model = Type1DCNNTransformer()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
MODEL_NAME = f'Type1DCNN_{scenario}_exp{EXP}'

# for classifier model
model_2nd = TypeClassifier()
model_2nd = model_2nd.to(device)
optimizer_2nd = torch.optim.Adam(model_2nd.parameters(), lr=1e-3, betas=(0.9, 0.999))
MODEL_NAME_2nd = f'TypeClassifier_{scenario}_exp{EXP}'


criterion = torch.nn.BCEWithLogitsLoss()

def train_one_epoch():
    running_loss = 0.
    batch_eers = []
    batch_losses = []
    for i, (input_data, labels) in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        optimizer_2nd.zero_grad()
        input_data[0] = Variable(input_data[0])
        input_data[1] = Variable(input_data[1])
        input_data[0] = input_data[0].to(device)
        input_data[1] = input_data[1].to(device)
        pred1 = model(input_data[0])
        pred2 = model(input_data[1])
        new_input = torch.stack((pred1, pred2), dim=2)
        pred = model_2nd(new_input)
        pred = pred.view(-1)
        loss0 = contrastive_loss(pred1, pred2, labels.to(torch.int64).double().to(device))
        loss = criterion(pred, labels.float().to(device))
        loss0.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer_2nd.step()
        running_loss += loss.item()
        pred1 = pred1.cpu().detach()
        pred2 = pred2.cpu().detach()
        dists = pred.cpu().detach()
        fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
        fnr = 1 - tpr
        EER1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        batch_eers.append(np.mean([EER1, EER2])*100)
        batch_losses.append(running_loss)
    return np.sum(batch_losses), np.mean(batch_eers)



loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []

best_v_eer = 100.


for epoch in range(EPOCHS):
    start = time.time()
    print('EPOCH:', epoch)
    model.eval()
    model_2nd.train()
    epoch_loss, epoch_eer = train_one_epoch()
    loss_t_list.append(epoch_loss)
    eer_t_list.append(epoch_eer)
    model_2nd.eval()
    running_loss_v = 0.
    with torch.no_grad():
        batch_eers_v = []
        batch_losses_v = []
        for i, (input_data, labels) in enumerate(val_dataloader, 0):
            input_data[0] = Variable(input_data[0])
            input_data[1] = Variable(input_data[1])
            input_data[0] = input_data[0].to(device)
            input_data[1] = input_data[1].to(device)
            pred1 = model(input_data[0])
            pred2 = model(input_data[1])
            new_input = torch.stack((pred1, pred2), dim=2)
            pred = model_2nd(new_input)
            pred = pred.view(-1)
            loss_v = criterion(pred, labels.float().to(device))
            running_loss_v += loss_v.item()
            pred1 = pred1.cpu().detach()
            pred2 = pred2.cpu().detach()
            dists = pred.cpu().detach()
            fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
            fnr = 1 - tpr
            EER1_v = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            EER2_v = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
            batch_eers_v.append(np.mean([EER1_v, EER2_v])*100)
            batch_losses_v.append(running_loss_v)
    eer_v_list.append(np.mean(batch_eers_v))
    loss_v_list.append(np.sum(batch_losses_v))

    print("Epoch Loss on Training Set: " + str(epoch_loss))
    print("Epoch Loss on Validation Set: " + str(np.sum(batch_losses_v)))
    print("Epoch EER [%] on Training Set: "+ str(epoch_eer))
    print("Epoch EER [%] on Validation Set: " + str(np.mean(batch_eers_v)))
    if eer_v_list[-1] < best_v_eer:
        torch.save(model.state_dict(), MODEL_NAME + '.pt')
        torch.save(model_2nd.state_dict(), MODEL_NAME_2nd + '.pt')
        print("New Best Epoch EER [%] on Validation Set: " + str(eer_v_list[-1]))
        best_v_eer = eer_v_list[-1]
    print("Best EER [%] on Validation Set So Far: " + str(best_v_eer))
    TRAIN_INFO_LOG_FILENAME = MODEL_NAME + '_log.txt'
    log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
    with open(TRAIN_INFO_LOG_FILENAME, "w") as output:
        output.write(str(log_list))
    end = time.time()
    print('Time for last epoch [min]: ' + str(np.round((end-start)/60, 2)))
