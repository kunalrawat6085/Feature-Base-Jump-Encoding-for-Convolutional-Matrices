# Feature-Base-Jump-Encoding-for-Convolutional-Matrices
Some theorising 

# Base-Jump Feature Encoding for Neural Network Matrices

This repository demonstrates a novel technique for feature encoding in neural network (NN) matrices.

## Summary

Instead of expanding matrix size (by adding channels) or increasing compute via multiplication, this method encodes the *presence of features* by incrementing matrix values by a fixed "base jump" whenever a feature is detected.  
This can help compress feature maps for low-memory environments or creative ML experiments.

- **Memory-efficient**: No new channels, shape stays fixed.
- **Lossless decoding**: Easily retrieve both the original value and number of feature jumps.
- **Simple logic**: Encode and decode with integer arithmetic.

## How it works

1. **Input:**  
    - `matrix` — original data (2D numpy array)
    - `feature_mask` — same shape, 1 where feature is detected, else 0
    - `base` — a value larger than any possible matrix entry

2. **Encode:**  
    - `encoded = matrix + (feature_mask * base)`

3. **Decode:**  
    - `jump_count = encoded // base`
    - `original = encoded % base`

## Example

```python
import numpy as np
from base_jump import base_jump_encode, base_jump_decode

matrix = np.array([[2, 3, 6],
                   [1, 4, 5]])
feature_mask = np.array([[1, 0, 0],
                         [0, 1, 0]])
base = 10

encoded = base_jump_encode(matrix, feature_mask, base)
print(encoded)
# Output: [[12  3  6]
#          [ 1 14  5]]

jumps, original = base_jump_decode(encoded, base)
print(jumps)     # [[1 0 0], [0 1 0]]
print(original)  # [[2 3 6], [1 4 5]]
