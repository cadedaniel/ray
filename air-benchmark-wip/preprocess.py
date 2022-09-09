#!/usr/bin/python3.9

import numpy as np

dataset_size = 100
image_dims = (1, 224, 224, 3)

# TODO need to download mnist and perform resnet50 preprocessing.
dataset = np.zeros((dataset_size, *image_dims, 1), dtype=np.int32)

np.save('preprocessed_data.npy', dataset)
