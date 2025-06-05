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
from sklearn.metrics import mean_squared_error, mean_absolute_error

from localization import utils
from localization.utils.constants import NUM_ROUTERS, NUM_SPECIAL_TOKENS, PADDING_IDX, SOS_IDX
from localization.models import RNNRegressorEmb
from localization.dataset import LangDataset


# def collate_fn(batch):
#     sequences, targets = zip(*batch)
#     lengths = torch.tensor([len(seq) for seq in sequences])
#     padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PADDING_IDX)
#     return padded_seqs, torch.stack(targets), lengths

# def collate_fn(batch):
#     sequences, targets = zip(*batch)
#     padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PADDING_IDX)
#     return padded_seqs, torch.stack(targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN Model')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-f', '--data_folder', type=str, default="data/generated", help='Data folder')
    parser.add_argument('-b','--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-n', '--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('-e', '--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('-hs', '--hidden_size', type=int, default=32, help='Hidden size')
    parser.add_argument('-s', '--src_dim', type=int, default=20, help='Source dimension')
    parser.add_argument('-o', '--out_dir', type=str, default="output/lrnn", help='Output folder')

    args = parser.parse_args()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    utils.make_deterministic(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_dataset = LangDataset(f"{args.data_folder}/trainingData_b0f0.pt", device,
                                args.src_dim, SOS_IDX, PADDING_IDX)
    val_dataset = LangDataset(f"{args.data_folder}/validationData_b0f0.pt", device,
                              args.src_dim, SOS_IDX, PADDING_IDX)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = RNNRegressorEmb(vocab_size=NUM_ROUTERS + NUM_SPECIAL_TOKENS,
                            embedding_dim=args.embedding_dim,
                            hidden_size=args.hidden_size,
                            padding_id=PADDING_IDX).to(device)
    min_loss = np.inf
    min_epoch = -1

    loss_function = nn.L1Loss()
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    if not path.exists(args.out_dir): makedirs(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = f"{args.out_dir}/trainer_{timestamp}"
    writer = SummaryWriter(out_folder)

    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}\n')

        # Training
        model.train()
        train_loss = 0
        for sequences, targets in tqdm(train_loader):
            outputs = model(sequences)
            loss = loss_function(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(sequences)

        train_loss /= len(train_dataset)
        print(f'Training loss: {train_loss}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in tqdm(val_loader):
                outputs = model(sequences)
                loss = loss_function(outputs, targets)
                val_loss += loss.item() * len(sequences)

        val_loss /= len(val_dataset)
        print(f'Validation loss: {val_loss}\n')

        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch+1)
        writer.add_scalar("Best Epoch", min_epoch+1, epoch+1)
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

    print('Training done!')
    writer.close()

    test_dataset = LangDataset("data/test.pt", device, args.src_dim, SOS_IDX, PADDING_IDX)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    y_pred = np.empty((0, 2))
    y_test = np.empty((0, 2))
    model.eval()
    with torch.no_grad():
        for sequences, targets in tqdm(test_loader):
            outputs = model(sequences)
            yp = outputs.cpu().numpy()
            yt = targets.cpu().numpy()
            y_pred = np.concatenate((y_pred, yp))
            y_test = np.concatenate((y_test, yt))

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)


    with open(f"{out_folder}/settings.txt", "w") as f:
        f.write(f"Source dim: {args.src_dim}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Embedding Dim: {args.embedding_dim}\n")
        f.write(f"Hidden Size: {args.hidden_size}\n")
        f.write(f"Num Epochs: {args.num_epochs}\n\n\n")
        f.write(f"Train RMSE: {train_loss:.2f}\n")
        f.write(f"Val RMSE: {val_loss:.2f}\n")
        f.write(f"Test RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}")

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
