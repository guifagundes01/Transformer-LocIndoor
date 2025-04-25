# Author: Guilherme Fagundes
from os import mkdir, path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from  localization.models.rbf_model import Model

class Augmentation:
    def __init__(self, model: Model, routers: list, seed: int = 33):
        """
        Initialize with a trained model (from rbf_model.py) to leverage power_probability_masks.

        Args:
            model: Trained instance of the Model class from rbf_model.py.
        """
        self.model = model
        self.routers = routers
        self.rng = np.random.default_rng(seed)

    def generate_augmented_data(self, num_samples_per_floor=100):
        """
        Generate augmented WiFi fingerprints using power_probability_masks and power_prior_probability_distribution.

        Args:
            num_samples_per_floor: Number of augmented samples to generate per floor.

        Returns:
            pd.DataFrame: augmented data with columns for building, floor, coordinates, and router powers.
        """
        augmented_data = []
        for building in self.model.power_probability_masks.keys():
            for floor in self.model.power_probability_masks[building].keys():
                routers = list(self.model.power_probability_masks[building][floor].keys())
                for _ in range(num_samples_per_floor):
                    # Sample a valid (x, y) location from the building's grid
                    valid_indices = np.arange(len(self.model.x_building[building]))
                    loc_idx = self.rng.choice(valid_indices)
                    x, y = self.model.x_building[building][loc_idx], self.model.y_building[building][loc_idx]
                    sample = {wap: 0.0 for wap in self.routers}

                    for router in routers:
                        # Get the probability distribution of powers for this router at (x, y)
                        power_probs = self.model.power_probability_masks[building][floor][router]
                        powers = list(power_probs.keys())
                        probs = [power_probs[p][loc_idx] * self.model.power_prior_probability_distribution[building][floor][router][p] * len(self.model.x_building[building]) for p in powers]  # Bayes
                        # probs = [power_probs[p][loc_idx] for p in powers]  # P(power | x, y)

                        probs = np.clip(probs, 0, None)  # Replace negatives with 0
                        epsilon = 1e-5
                        probs += epsilon
                        probs = probs / np.sum(probs)     # Renormalize

                        # Sample a power value
                        power = self.rng.choice(powers, p=probs)
                        sample[router] = power

                        sample.update({
                            'x': x,
                            'y': y,
                            'FLOOR': floor,
                            'BUILDINGID': building,
                            'SPACEID': -1,
                            'RELATIVEPOSITION': -1,
                            'USERID': -1,
                            'PHONEID': -1,
                            'TIMESTAMP': -1,
                        })
                    augmented_data.append(sample)

    def generate_augmented_data_batched(self, num_samples_per_floor=100, batch_size=10, data_dir="data/generated"):
        """
        Generate augmented WiFi fingerprints using power_probability_masks and power_prior_probability_distribution.

        Args:
            num_samples_per_floor: Number of augmented samples to generate per floor.
            batch_size: Number of samples to generate in each batch.
            data_dir: Directory to save the generated data.

        Returns:
            pd.DataFrame: augmented data with columns for building, floor, coordinates, and router powers.
        """

        num_batches = num_samples_per_floor // batch_size
        y_columns = ["LATITUDE", "LONGITUDE", "x", "y"]
        metadata_columns = ["BUILDINGID", "FLOOR", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
        generated_data = pd.DataFrame(columns=self.routers + y_columns + metadata_columns)
        if not path.exists(data_dir): mkdir(data_dir)
        generated_data.to_csv(f'{data_dir}/trainingData.csv', index=False)
        generated_data.to_csv(f'{data_dir}/validationData.csv', index=False)

        bs = [b for b, nf in self.model.num_floors_in_each_building.items() for _ in range(nf * batch_size)]
        fs = [f for _, nf in self.model.num_floors_in_each_building.items() for f in range(nf) for _ in range(batch_size)]
        for _ in tqdm(range(num_batches)):
            augmented_data = []

            for i in range(batch_size):
                routers = list(self.model.power_probability_masks[bs[i]][fs[i]].keys())
                # Sample a valid (x, y) location from the building's grid
                valid_indices = np.arange(len(self.model.x_building[bs[i]]))
                loc_idx = self.rng.choice(valid_indices)
                x, y = self.model.x_building[bs[i]][loc_idx], self.model.y_building[bs[i]][loc_idx]
                sample = {wap: 0.0 for wap in self.routers}

                for router in routers:
                    # Get the probability distribution of powers for this router at (x, y)
                    p_xy_given_bfrp = self.model.power_probability_masks[bs[i]][fs[i]][router]
                    p_p = self.model.power_prior_probability_distribution[bs[i]][fs[i]][router]
                    powers = list(p_xy_given_bfrp.keys())
                    probs = [p_xy_given_bfrp[p][loc_idx] * p_p[p] * len(self.model.x_building[bs[i]]) for p in powers]  # Bayes
                    # probs = [power_probs[p][loc_idx] for p in powers]  # P(power | x, y)

                    probs = np.clip(probs, 0, None)  # Replace negatives with 0
                    epsilon = 1e-5
                    probs += epsilon
                    probs = probs / np.sum(probs)     # Renormalize

                    # Sample a power value
                    power = self.rng.choice(powers, p=probs)
                    sample[router] = power

                    sample.update({
                        'x': x,
                        'y': y,
                        'FLOOR': fs[i],
                        'BUILDINGID': bs[i],
                        'SPACEID': -1,
                        'RELATIVEPOSITION': -1,
                        'USERID': -1,
                        'PHONEID': -1,
                        'TIMESTAMP': -1,
                    })
                augmented_data.append(sample)

            augmented_data = pd.DataFrame(augmented_data)
            train_df, test_df = train_test_split(augmented_data, test_size=0.1)

            train_df.to_csv(f'{data_dir}/trainingData.csv', index=False, mode='a', header=False)
            test_df.to_csv(f'{data_dir}/validationData.csv', index=False, mode='a', header=False)

    def generate_batched_augmented_data_w_parameters(self, num_samples=100, batch_size=10, data_dir='data/generated'):
        """
        Generate augmented WiFi fingerprints using mu and phi rbf models.

        Args:
            num_samples: Number of augmented samples to generate.
            batch_size: Number of samples to generate in each batch.
            data_dir: Directory to save the generated data.

        Returns:
            pd.DataFrame: augmented data with columns for building, floor, coordinates, and router powers.
        """
        num_batches = num_samples // batch_size
        y_columns = ["LATITUDE", "LONGITUDE", "x", "y"]
        metadata_columns = ["BUILDINGID", "FLOOR", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
        generated_data = pd.DataFrame(columns=self.routers + y_columns + metadata_columns)
        if not path.exists(data_dir): mkdir(data_dir)
        generated_data.to_csv(f'{data_dir}/trainingData.csv', index=False)
        generated_data.to_csv(f'{data_dir}/validationData.csv', index=False)

        for _ in tqdm(range(num_batches)):
            b = self.rng.integers(0, self.model.num_buildings, size=batch_size)
            f = np.array([self.rng.integers(0, self.model.num_floors_in_each_building[building]) for building in b])
            idxs = [self.rng.integers(0, len(self.model.x_building[building])) for building in b]
            x = np.array([self.model.x_building[building][idx] for (building, idx) in zip(b, idxs)])
            y = np.array([self.model.y_building[building][idx] for (building, idx) in zip(b, idxs)])
            # x = np.array([self.rng.uniform(self.model.x_building[building][0], self.model.x_building[building][-1]) for building in b])
            # y = np.array([self.rng.uniform(self.model.y_building[building][0], self.model.y_building[building][-1]) for building in b])

            gen_data = {r: [] for r in self.routers}
            for i in range(batch_size):
                for r in self.routers:
                    if r in self.model.mu_rbf[b[i]][f[i]]:
                        mu = self.model.mu_rbf[b[i]][f[i]][r](x[i], y[i])
                        phi = np.minimum(self.model.phi_rbf[b[i]][f[i]][r](x[i], y[i]), 1)
                        gen_data[r].append(self.rng.normal(mu, self.model.sigma) * self.rng.binomial(1, phi))
                    else:
                        gen_data[r].append(np.nan)

            for r in gen_data:
                gen_data[r] = np.array(gen_data[r])

            gen_data.update({
                "BUILDINGID": b,
                "FLOOR": f,
                "SPACEID": -1 * np.ones(batch_size),
                "RELATIVEPOSITION": -1 * np.ones(batch_size),
                "USERID": -1 * np.ones(batch_size),
                "PHONEID": -1 * np.ones(batch_size),
                "TIMESTAMP": -1 * np.ones(batch_size),
                "LATITUDE": -1 * np.ones(batch_size),
                "LONGITUDE": -1 * np.ones(batch_size),
                "x": x,
                "y": y
            })
            generated_data = pd.DataFrame(gen_data)
            generated_data = generated_data.fillna(0.0)
            generated_data = generated_data.clip(lower=0)
            train_df, test_df = train_test_split(generated_data, test_size=0.1)

            train_df.to_csv(f'{data_dir}/trainingData.csv', index=False, mode='a', header=False)
            test_df.to_csv(f'{data_dir}/validationData.csv', index=False, mode='a', header=False)

    def generate_augmented_data_w_parameters(self, num_samples=100):
        """
        Generate augmented WiFi fingerprints using mu and phi rbf models.

        Args:
            num_samples: Number of augmented samples to generate.

        Returns:
            pd.DataFrame: augmented data with columns for building, floor, coordinates, and router powers.
        """
        b = self.rng.integers(0, self.model.num_buildings, size=num_samples)
        f = np.array([self.rng.integers(0, self.model.num_floors_in_each_building[building]) for building in b])
        idxs = [self.rng.integers(0, len(self.model.x_building[building])) for building in b]
        x = self.model.x_building[[b, idxs]]
        y = self.model.y_building[[b, idxs]]
        # x = np.array([rng.uniform(model.x_building[building][0], model.x_building[building][-1]) for building in b])
        # y = np.array([rng.uniform(model.y_building[building][0], model.y_building[building][-1]) for building in b])

        gen_data = {r: [] for r in self.routers}
        for i in range(num_samples):
            for r in self.routers:
                if r in self.model.mu_rbf[b[i]][f[i]]:
                    mu = self.model.mu_rbf[b[i]][f[i]][r](x[i], y[i])
                    phi = np.minimum(self.model.phi_rbf[b[i]][f[i]][r](x[i], y[i]), 1)
                    gen_data[r].append(self.rng.normal(mu, self.model.sigma) * self.rng.binomial(1, phi))
                else:
                    gen_data[r].append(np.nan)

        for r in self.routers:
            gen_data[r] = np.array(gen_data[r])

        gen_data["BUILDINGID"] = b
        gen_data["FLOOR"] = f
        gen_data["SPACEID"] = -np.ones(num_samples)
        gen_data["RELATIVEPOSITION"] = -np.ones(num_samples)
        gen_data["USERID"] = -np.ones(num_samples)
        gen_data["PHONEID"] = -np.ones(num_samples)
        gen_data["TIMESTAMP"] = -np.ones(num_samples)
        gen_data["LATITUDE"] = -np.ones(num_samples)
        gen_data["LONGITUDE"] = -np.ones(num_samples)
        gen_data["x"] = x
        gen_data["y"] = y
        generated_data = pd.DataFrame(gen_data)

        return generated_data
