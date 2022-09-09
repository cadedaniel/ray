#!/usr/bin/python3.9

import numpy as np
import time

dataset_size = 100
image_dims = (1, 224, 224, 3)

dataset = np.zeros((dataset_size, *image_dims, 1), dtype=np.int32)
#labels = np.zeros((dataset_size, 1), dtype=np.int32)

np.save('preprocessed_data.npy', dataset)
