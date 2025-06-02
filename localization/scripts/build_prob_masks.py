import argparse
import pickle
import json
import os

import numpy as np
from tqdm import trange
from scipy.interpolate import Rbf

from localization import utils


def radial_log_basis_function(model, r):
    return np.log(r + model.epsilon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Propability Masks')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-s','--size', type=int, default=100, help='Grid size')
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    utils.make_deterministic(args.seed)

    data_dir = f"data/probs-{args.size}/"
    num_buildings = 3
    num_floors_in_each_building = {0: 4,
                                    1: 4,
                                    2: 5}
    Rbf.radial_log_basis_function = radial_log_basis_function

    os.makedirs(data_dir, exist_ok=True)
    for building in range(num_buildings):
        for floor in trange(num_floors_in_each_building[building]):

            with open(f'output/filtered_model_{args.size}.bin', 'rb') as inp:
                model = pickle.load(inp)
            routers = list(model.power_probability_masks[building][floor].keys())
            x_size = len(model.x_building[building])
            valid_indices = np.arange(x_size)
            power_probability_masks = model.power_probability_masks[building][floor]
            power_prior_probability_distribution = model.power_prior_probability_distribution[building][floor]
            del model

            power_distribution = []
            for idx in valid_indices:
                router_distribution = {}
                for router in routers:
                    # Get the probability distribution of powers for this router at (x, y)
                    p_xy_given_bfrp = power_probability_masks[router]
                    p_p = power_prior_probability_distribution[router]
                    powers = list(p_xy_given_bfrp.keys())
                    probs = np.array([p_xy_given_bfrp[p][idx] * p_p[p] * x_size for p in powers], np.float64)  # Bayes
                    # probs = [power_probs[p][loc_idx] for p in powers]  # P(power | x, y)

                    probs = np.clip(probs, 0, None)  # Replace negatives with 0
                    epsilon = 1e-5
                    probs += epsilon
                    probs = probs / np.sum(probs)     # Renormalize
                    router_distribution[router] = probs.tolist()
                power_distribution.append(router_distribution)

            with open(f"{data_dir}/power_distribution_building_{building}_floor_{floor}.json", "w") as f:
                json.dump(power_distribution, f)
