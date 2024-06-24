using ExplainableAI: augment_batch_dim, augment_indices, reduce_augmentation
using ExplainableAI: interpolate_batch
using Test

# augment_batch_dim
A = [1 2; 3 4]
B = @inferred augment_batch_dim(A, 3)
@test B == [
    1 1 1 2 2 2
    3 3 3 4 4 4
]
B = @inferred augment_batch_dim(A, 4)
@test B == [
    1 1 1 1 2 2 2 2
    3 3 3 3 4 4 4 4
]
A = reshape(1:8, 2, 2, 2)
B = @inferred augment_batch_dim(A, 3)
@test B[:, :, 1] == A[:, :, 1]
@test B[:, :, 2] == A[:, :, 1]
@test B[:, :, 3] == A[:, :, 1]
@test B[:, :, 4] == A[:, :, 2]
@test B[:, :, 5] == A[:, :, 2]
@test B[:, :, 6] == A[:, :, 2]

# augment_batch_dim
inds = [CartesianIndex(5, 1), CartesianIndex(3, 2)]
augmented_inds = @inferred augment_indices(inds, 3)
@test augmented_inds == [
    CartesianIndex(5, 1)
    CartesianIndex(5, 2)
    CartesianIndex(5, 3)
    CartesianIndex(3, 4)
    CartesianIndex(3, 5)
    CartesianIndex(3, 6)
]

# reduce_augmentation
A = Float32.(reshape(1:10, 1, 1, 10))
R = @inferred reduce_augmentation(A, 5)
@test R == reshape([sum(1:5), sum(6:10)] / 5, 1, 1, :)
A = Float64.(reshape(1:10, 1, 1, 1, 1, 10))
R = @inferred reduce_augmentation(A, 2)
@test R == reshape([3, 7, 11, 15, 19] / 2, 1, 1, 1, 1, :)

x = Float16.(reshape(1:4, 2, 2))
x0 = zero(x)
A = @inferred interpolate_batch(x, x0, 5)
@test A ≈ [
    0.0 0.25 0.5 0.75 1.0 0.0 0.75 1.5 2.25 3.0
    0.0 0.5  1.0 1.5  2.0 0.0 1.0  2.0 3.0  4.0
]
