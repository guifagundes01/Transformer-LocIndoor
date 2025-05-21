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

from localization import utils
from localization.utils.constants import NUM_ROUTERS, NUM_SPECIAL_TOKENS, PADDING_IDX, SOS_IDX
from localization.models import RNNRegressorEmb
from localization.dataset import LangMaskDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN Model')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-b','--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('-p', '--prob', type=float, default=0.4, help='Probability of add a wrong router')
    parser.add_argument('-s', '--src_dim', type=int, default=20, help='Source dimension')
    parser.add_argument('-f', '--data_folder', type=str, default="data/generated", help='Data folder')
    parser.add_argument('-o', '--out_dir', type=str, default="output/rnn", help='Output folder')

    args = parser.parse_args()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    utils.make_deterministic(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_dataset = LangMaskDataset(f"{args.data_folder}/trainingDataL.pt", device,
                                    args.src_dim, NUM_ROUTERS+1, SOS_IDX, PADDING_IDX, args.prob)
    val_dataset = LangMaskDataset(f"{args.data_folder}/validationDataL.pt", device,
                                  args.src_dim, NUM_ROUTERS+1, SOS_IDX, PADDING_IDX, args.prob)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = RNNRegressorEmb(vocab_size=NUM_ROUTERS + NUM_SPECIAL_TOKENS,
                            embedding_dim=64, hidden_size=256).to(device)
    min_loss = np.inf
    min_epoch = -1
    patience_counter = 0

    # loss_function = nn.L1Loss()
    loss_function = nn.CrossEntropyLoss()
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
        for sequences, targets in tqdm(train_loader):
            logits = model(sequences)
            loss = loss_function(logits, targets)

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
                logits = model(sequences)
                loss = loss_function(logits, targets)
                val_loss += loss.item() * len(sequences)

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
