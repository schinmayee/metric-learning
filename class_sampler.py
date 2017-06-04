import numpy as np
import torch

class ClassSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data, num_species, num_per_specie, num_batches):
        self.data = data
        self.labels = data.labels
        self.species = np.unique(self.labels)
        self.num_batches = num_batches

        # per batch sampling variables
        self.num_species = num_species
        self.num_per_specie = num_per_specie

        self.len = num_batches*num_species*num_per_specie

    def __iter__(self):
        indices = list()
        for b in range(self.num_batches):
            this_batch_species = np.random.choice(self.species, self.num_species,
                                                  replace=False)
            for s in this_batch_species:
                sel = np.where(self.labels == s)[0]
                sel = np.random.choice(sel, self.num_per_specie)
                indices = indices + list(sel)
        return iter(indices)

    def __len__(self):
        return self.len
