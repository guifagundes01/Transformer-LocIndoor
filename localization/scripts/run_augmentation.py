import pickle
import argparse

import numpy as np
from scipy.interpolate import Rbf

from localization import dataset
from localization.augmentation.augmentation_rbf_model import Augmentation
from localization.models.rbf_model import Model

def radial_log_basis_function(self, r):
    return np.log(r + self.epsilon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tLoc')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-n', '--num_samples', type=int, default=1620, help='Number of samples per floor')
    parser.add_argument('-bs','--batch_size', type=int, default=180, help='Batch Size')
    parser.add_argument('-t', '--test_size', type=float, default=0.06, help='Size of the test dataset')
    parser.add_argument('-df', '--data_folder', type=str, default="data/generated", help='Folder to save the generated data')
    parser.add_argument('-b','--building', type=int, default=None, help='Building number')
    parser.add_argument('-f','--floor', type=int, default=None, help='Floor number')
    args = parser.parse_args()

    data = dataset.load_ujiindoor_loc("data/filtered")

    Rbf.radial_log_basis_function = radial_log_basis_function
    with open('output/filtered_model.bin', 'rb') as inp:
        model: Model = pickle.load(inp)

    routers = model.get_all_routers_in_this_floor(data.get_full_df().columns)
    augmenter = Augmentation(model, routers)
    if args.building is None or args.floor is None:
        augmenter.generate_augmented_data_batched(num_samples_per_floor=args.num_samples,
                                                  batch_size=args.batch_size, test_size=args.test_size,
                                                  data_dir=args.data_folder)
    else:
        augmenter.generate_augmented_data_batched_for_bf(num_samples_per_floor=args.num_samples,
                                                         batch_size=args.batch_size, test_size=args.test_size,
                                                         data_dir=args.data_folder, building=args.building,
                                                         floor=args.floor)
