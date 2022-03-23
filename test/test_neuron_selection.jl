using ExplainableAI: MaxActivationSelector, IndexSelector
using Random

ns_max = @inferred MaxActivationSelector()
ns_idx = @inferred IndexSelector(4)

# Array
A = [-2.1694243, 2.4023275, 0.99464744, -0.1514646, 1.0307171]
R = similar(A)

@test_throws ArgumentError @inferred ns_max(A)
@test_throws ArgumentError @inferred ns_idx(A)

# 5x1 input
A = reshape(A, 5, 1)
R = similar(A)

I_max = @inferred ns_max(A)
I_idx = @inferred ns_idx(A)
@test I_max isa Vector{CartesianIndex{2}}
@test I_idx isa Vector{CartesianIndex{2}}
@test I_max == [CartesianIndex(2, 1)]
@test I_idx == [CartesianIndex(4, 1)]

# 5x3 input
A = [
    0.0564584 0.736398 0.370134
    0.112634 0.0193744 0.679662
    0.942529 0.335366 0.0132769
    0.170132 0.474743 0.590655
    0.218707 0.0440574 0.962128
]
R = similar(A)

I_max = @inferred ns_max(A)
I_idx = @inferred ns_idx(A)
@test I_max isa Vector{CartesianIndex{2}}
@test I_idx isa Vector{CartesianIndex{2}}
@test I_max == [
    CartesianIndex(3, 1)
    CartesianIndex(1, 2)
    CartesianIndex(5, 3)
]
@test I_idx == [
    CartesianIndex(4, 1)
    CartesianIndex(4, 2)
    CartesianIndex(4, 3)
]

# 4x3x2 input with Tuple IndexSelector
ns_idx = @inferred IndexSelector((4, 2))
A = rand(MersenneTwister(1234), Float32, 4, 3, 2)
R = similar(A)

I_max = @inferred ns_max(A)
I_idx = @inferred ns_idx(A)
@test I_max isa Vector{CartesianIndex{3}}
@test I_idx isa Vector{CartesianIndex{3}}
@test I_max == [
    CartesianIndex(2, 1, 1)
    CartesianIndex(4, 1, 2)
]
@test I_idx == [
    CartesianIndex(4, 2, 1)
    CartesianIndex(4, 2, 2)
]
