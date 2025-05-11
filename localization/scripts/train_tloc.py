# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pickle

from localization.models.rbf_model import Model
from localization import dataset, utils
from os import path, mkdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tLoc')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('-s','--data_size', type=int, default=1, help='Size of the training data')
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    utils.make_deterministic(args.seed)
    num_buildings = 3
    num_floors_in_each_building = {0: 4,
                                   1: 4,
                                   2: 5}
    path_to_the_polygon_files_of_each_building = {0: 'data/geometry_building_0.npy',
                                                  1: 'data/geometry_building_1.npy',
                                                  2: 'data/geometry_building_2.npy'}

    sigma = 6.75
    num_samples_per_ap = 20

    dataset = dataset.load_ujiindoor_loc(data_folder=f'data/generated-250/sample-{args.data_size}x', transform=False)

    model = Model(num_buildings,
                  num_floors_in_each_building,
                  path_to_the_polygon_files_of_each_building,
                  sigma,
                  num_samples_per_ap)
    model.train(dataset)

    if not path.exists('output'): mkdir('output')

    with open(f'output/model_gen_250_{args.data_size}x.bin', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
