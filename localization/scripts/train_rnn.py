import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from localization.models.rnn import Model
from localization import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN Model')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-b','--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('-p', '--patience', type=int, default=20, help='Patience')
    parser.add_argument('-f', '--data_folder', type=str, default="data/generated", help='Data folder')
    parser.add_argument('-d', '--out_dir', type=str, default="output/rnn", help='Output folder')

    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    utils.make_deterministic(args.seed)
    train_x = pd.read_csv(args.data_folder + "/trainingData.csv")
    val_x = pd.read_csv(args.data_folder + "/validationData.csv")
    columns_to_drop = ["x", "y", "BUILDIINGID", "FLOOR"]
    y_columns = ["x", "y"]
    train_y = train_x[y_columns].to_numpy()
    val_y = val_x[y_columns].to_numpy()
    train_x = train_x.drop(columns=columns_to_drop).to_numpy()
    val_x = val_x.drop(columns=columns_to_drop).to_numpy()

    print(train_x.shape, train_y.shape)
    print(val_x.shape, val_y.shape)

    train_dataset = RSSDataset(train_x, train_y)
    val_dataset = RSSDataset(val_x, val_y)

    # weights = torch.Tensor(train_y.sum() / train_y.sum(axis=0)).to(device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = Model(train_x.shape[1], 256, 2)
    train_loss_history = []
    val_loss_history = []
    min_loss = np.inf
    min_epoch = -1
    patience_counter = 0

    # loss_function = nn.L1Loss()
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}\n')

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

        with open(os.path.join(args.out_dir, 'log.txt'), 'a') as f:
            f.write(f"Epoch: {epoch+1}/{args.num_epochs}\n")
            f.write(f"      Training loss: {train_loss_history[-1]}\n")
            f.write(f"      Validation loss: {val_loss_history[-1]}\n")
            f.write(f"      Best epoch / loss: {min_epoch+1} / {min_loss}\n")

        # model saving
        if min_loss > val_loss_history[-1]:

            # update best loss
            min_epoch = epoch
            min_loss = val_loss_history[-1]

            # save model
            model_path = os.path.join(args.out_dir, 'rnn_model.pth')
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, model_path)

        # early stopping
        if len(val_loss_history) >= 2:

            if val_loss_history[-1] > min_loss:
              patience_counter+=1
            else:
              patience_counter = 0

        if patience_counter >= args.patience:
            print('Training done!')
            break
