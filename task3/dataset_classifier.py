import numpy as np
from load_data import CTDataset1

class TrainingSet(CTDataset1):
    def __init__(self, config, mode, horizontal_only=True):
        super().__init__(config, mode, horizontal_only)
        self.organs = np.array(self.organs) - 1
        
    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        mask = self.total_labels[idx]
        img = self.total_images[idx]
        mask = np.expand_dims(mask, axis=0)
        img = np.expand_dims(img, axis=0)
        new_img = np.concatenate([mask, img], axis=0)
        return mask, self.organs[idx]
    
    