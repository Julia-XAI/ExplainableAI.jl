using Flux
using Flux: flatten
using ExplainableAI: drop_batch_index, masked_copy
using Random

# drop_batch_index
I1 = CartesianIndex(5, 3, 2)
I2 = @inferred drop_batch_index(I1)
@test I2 == CartesianIndex(5, 3)
I1 = CartesianIndex(5, 3, 2, 6)
I2 = @inferred drop_batch_index(I1)
@test I2 == CartesianIndex(5, 3, 2)

# masked_copy
A    = [4  9  9; 9  6  9; 1  7  8]
mask = Matrix{Bool}([0  1  1; 0  1  0; 1  1  1])
mc   = @inferred masked_copy(A, mask)
@test mc == [0  9  9; 0  6  0; 1  7  8]
