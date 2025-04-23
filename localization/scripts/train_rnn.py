import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from localization.models.rnn import Model
from localization import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Parameters
batch_size = 128
num_epochs = 100
learning_rate = 1e-2
patience = 20
out_dir = "output/rnn/"

data = dataset.load_ujiindoor_loc(data_folder='data/generated')

class RSSDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index]).to(device)
        y = torch.Tensor(self.y[index]).to(device)
        return (x, y)

    def __len__(self):
        count = self.x.shape[0]
        return count

train_x, val_x = data.get_X()
train_yn, val_yn = data.get_normalized_y()
train_yc, val_yc = data.get_categorical_y()
train_y = np.hstack([train_yn, train_yc])
val_y = np.hstack([val_yn, val_yc])

train_dataset = RSSDataset(train_x, train_y)
val_dataset = RSSDataset(val_x, val_y)

# weights = torch.Tensor(train_y.sum() / train_y.sum(axis=0)).to(device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Model(train_x.shape[1], 256)
train_loss_history = []
val_loss_history = []
min_loss = np.inf
min_epoch = -1
patience_counter = 0

# loss_function = nn.L1Loss()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print('Epoch %d/%d\n' % (epoch+1, num_epochs))

    # Training
    model.train()
    train_loss = []

    for batch_idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item()*len(data))

    train_loss_history.append(np.sum(train_loss) / len(train_dataset))
    print('Training loss: {}'.format(train_loss_history[-1]))

    # Validation
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            outputs = model(data)
            loss = loss_function(outputs, labels)
            val_loss.append(loss.item()*len(data))

        val_loss_history.append(np.sum(val_loss) / len(val_dataset))
        print('Validation loss: {}\n'.format(val_loss_history[-1]))

    with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
        f.write("Epoch: {}/{}\n".format(epoch+1, num_epochs))
        f.write("      Training loss: {}\n".format(train_loss_history[-1]))
        f.write("      Validation loss: {}\n".format(val_loss_history[-1]))
        f.write("      Best epoch / loss: {} / {}\n".format(min_epoch+1, min_loss))

    # model saving
    if min_loss > val_loss_history[-1]:

        # update best loss
        min_epoch = epoch
        min_loss = val_loss_history[-1]

        # save model
        model_path = os.path.join(out_dir, 'rnn_model.pth')
        best_state_dict = model.state_dict()
        torch.save(best_state_dict, model_path)

    # early stopping
    if len(val_loss_history) >= 2:

        if val_loss_history[-1] > min_loss:
          patience_counter+=1
        else:
          patience_counter = 0

    if patience_counter >= patience:
        print('Training done!')
        break
