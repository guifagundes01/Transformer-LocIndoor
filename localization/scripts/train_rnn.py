import argparse

from os import path, makedirs
from datetime import datetime

import numpy as np
import torch

from tqdm import tqdm
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from localization.models import RNNRegressor
from localization.dataset import RSSDataset
from localization import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN Model')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-b','--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('-p', '--patience', type=int, default=15, help='Patience')
    parser.add_argument('-f', '--data_folder', type=str, default="data/generated", help='Data folder')
    parser.add_argument('-d', '--out_dir', type=str, default="output/rnn", help='Output folder')
    parser.add_argument('-r', '--r_path', type=str, default="data/routers.npy", help='File to used routers')

    args = parser.parse_args()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    utils.make_deterministic(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = RSSDataset(f"{args.data_folder}/trainingData.csv", device)
    val_dataset = RSSDataset(f"{args.data_folder}/validationData.csv", device, train_dataset.routers)

    # weights = torch.Tensor(train_y.sum() / train_y.sum(axis=0)).to(device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = RNNRegressor(len(train_dataset.routers), 256, 2).to(device)
    min_loss = np.inf
    min_epoch = -1
    patience_counter = 0

    loss_function = nn.L1Loss()
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    if not path.exists(args.out_dir): makedirs(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{args.out_dir}/trainer_{timestamp}")

    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}\n')

        # Training
        model.train()
        train_loss = 0
        for data, labels in tqdm(train_loader):
            outputs = model(data)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(data)

        train_loss /= len(train_dataset)
        print(f'Training loss: {train_loss}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in tqdm(val_loader):
                outputs = model(data)
                loss = loss_function(outputs, labels)
                val_loss += loss.item() * len(data)

        val_loss /= len(val_dataset)
        print(f'Validation loss: {val_loss}\n')

        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalar("Best Epoch", min_epoch+1, epoch)
        writer.flush()

        # model saving
        if min_loss > val_loss:
            # update best loss
            min_epoch = epoch
            min_loss = val_loss

            # save model
            model_path = path.join(args.out_dir, 'rnn_model.pth')
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, model_path)

        # early stopping
        if epoch+1 >= 2:
            if val_loss > min_loss:
              patience_counter += 1
            else:
              patience_counter = 0

        if patience_counter >= args.patience:
            print('Training done!')
            break

    writer.close()
