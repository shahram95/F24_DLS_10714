from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip 
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                    raise ValueError("Invalid magic number in image file")
            
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, rows * cols)
            X = images.astype(np.float32) / 255.0

            # Read labels

        with gzip.open(label_filename, 'rb') as f:
                magic, num_labels = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError("Invalid magic number in label file")
                
                label_data = f.read()
                y = np.frombuffer(label_data, dtype=np.uint8)
        
            # Check consistency
        if num_images != num_labels:
            raise ValueError(f"Number of images ({num_images}) and labels ({num_labels}) do not match")
        
        self.imgs = X
        self.labels = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image, label = self.imgs[index], self.labels[index]

        if self.transforms:
            image = image.reshape(28, 28, 1)
            image = self.apply_transforms(image)
            image = image.reshape(-1)
        
        return image, label
            
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.imgs)
        ### END YOUR SOLUTION