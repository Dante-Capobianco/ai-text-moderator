import numpy as np

#Custom layer normalization - unused for computational efficiency & lack of learnable parameters compared to TensorFlow
def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)