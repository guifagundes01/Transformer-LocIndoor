# Author: Guilherme Fagundes
import numpy as np
import pandas as pd

class Augmentation:
    def __init__(self, model, routers):
        """
        Initialize with a trained model (from rbf_model.py) to leverage power_probability_masks.
        
        Args:
            model: Trained instance of the Model class from rbf_model.py.
        """
        self.model = model

        self.routers = routers

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
                    loc_idx = np.random.choice(valid_indices)
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
                        power = np.random.choice(powers, p=probs)
                        sample[router] = power

                        sample.update({
                            'x': x,
                            'y': y,
                            'FLOOR': floor,
                            'BUILDINGID': building,
                            'SPACEID': 0,   
                            'RELATIVEPOSITION': 0,
                            'USERID': 0,
                            'PHONEID': 0,
                            'TIMESTAMP': 0,
                        })
                        
                    
                    augmented_data.append(sample)
        
        return pd.DataFrame(augmented_data)
    



