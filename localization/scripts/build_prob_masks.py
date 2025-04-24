import argparse
import pickle

import numpy as np
from tqdm import tqdm
from scipy.interpolate import Rbf

from localization.models.rbf_model import Model
from localization import utils, dataset


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

    Rbf.radial_log_basis_function = radial_log_basis_function
    with open('output/filtered_model.bin', 'rb') as inp:
        model: Model = pickle.load(inp)

    grid_size = args.size
    data_dir = f"data/generated-{grid_size}/"


    data = dataset.load_ujiindoor_loc(data_folder='data/filtered')
    model.grid_size_in_each_dimension_in_each_building = grid_size
    for building in range(model.num_buildings):
        x_building, y_building = model.construct_building_map(data, building)
        model.x_building[building] = x_building
        model.y_building[building] = y_building

        for floor in tqdm(range(model.num_floors_in_each_building[building])):
            filtered_dataset = data.get_floor_data(building=building, floor=floor)
            X_train = filtered_dataset.get_full_df()
            routers = model.get_all_routers_in_this_floor(X_train.columns)

            for router in routers:
                if model.checking_non_null_minimum_percentage_of_samples(X_train, router):
                    model.power_probability_masks[building][floor][router] = {}
                    mu_building = model.mu_rbf[building][floor][router](x_building, y_building)
                    phi_building =  model.phi_rbf[building][floor][router](x_building, y_building)

                    for power in range(0, model.max_power):
                        model.power_probability_masks[building][floor][router][
                            power] = model.approximate_position_density_function_given_router(power, mu_building,
                                                                                             phi_building)

    model_data = {
        "x_building": model.x_building,
        "y_building": model.y_building,
        "power_probability_masks": model.power_probability_masks,
        "power_prior_probability_distribution": model.power_prior_probability_distribution,
    }

    with open(f'output/model_data_filtered_{grid_size}.bin', 'wb') as outp:
        pickle.dump(model_data, outp, pickle.HIGHEST_PROTOCOL)

    # with open(f'output/model_filtered_{grid_size}.bin', 'wb') as outp:
    #     pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
