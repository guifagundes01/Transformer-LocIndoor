from localization.augmentation.augmentation_rbf_model import Augmentation
import pickle
from os import path, mkdir
import pandas as pd
from localization.dataset.ujiindoorloc import calculate_means

def train_test_split_random(df, test_size=0.5):
    """Randomly split DataFrame into train/test sets."""
    val_df = df[df.shape[0] // 2:]
    train_df = df[:df.shape[0] // 2]   
    return train_df, val_df

# Load trained model
with open('output/filtered_model.bin', 'rb') as inp:
    model = pickle.load(inp)

df = pd.read_csv('data/filtered/validationData.csv')

routers = df.columns[:-9]

# Generate augmented data
augmenter = Augmentation(model, routers)
augmented_df = augmenter.generate_augmented_data(num_samples_per_floor=1000)

if not path.exists('data/augmented_bayes'): mkdir('data/augmented_bayes')

train_df, val_df = train_test_split_random(augmented_df)

# Save augmented data
train_df.to_csv("data/augmented_bayes/trainingData.csv", index=False)
val_df.to_csv("data/augmented_bayes/validationData.csv", index=False)