import pickle
import argparse

import numpy as np
from scipy.interpolate import Rbf

from localization import dataset
from localization.augmentation.augmentation_rbf_model import Augmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tLoc')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-n', '--num_samples', type=int, default=1620, help='Number of samples per floor')
    parser.add_argument('-b','--batch_size', type=int, default=180, help='Batch Size')
    parser.add_argument('-t', '--test_size', type=float, default=0.06, help='Size of the test dataset')
    args = parser.parse_args()

    data = dataset.load_ujiindoor_loc("data/filtered")

    def radial_log_basis_function(self, r):
        return np.log(r + self.epsilon)

    Rbf.radial_log_basis_function = radial_log_basis_function
    with open('output/filtered_model.bin', 'rb') as inp:
        model = pickle.load(inp)

    routers = model.get_all_routers_in_this_floor(data.get_full_df().columns)
    augmenter = Augmentation(model, routers)
    augmenter.generate_augmented_data_batched(num_samples_per_floor=args.num_samples,
                                              batch_size=args.batch_size, test_size=args.test_size,
                                              data_dir="data/generated")
