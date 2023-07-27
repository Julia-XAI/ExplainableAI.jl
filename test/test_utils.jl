using Flux
using Flux: flatten
using ExplainableAI: flatten_model
using ExplainableAI: has_output_softmax, check_output_softmax, activation_fn
using ExplainableAI: stabilize_denom, batch_dim_view, drop_batch_index, masked_copy
using Random

pseudorand(dims...) = rand(MersenneTwister(123), Float32, dims...)

# Test `activation_fn`
@test activation_fn(Dense(5, 2, gelu)) == gelu
@test activation_fn(Conv((5, 5), 3 => 2, softplus)) == softplus
@test activation_fn(BatchNorm(5, selu)) == selu
@test isnothing(activation_fn(flatten))

# flatten_model
@test flatten_model(Chain(Chain(Chain(abs)), sqrt, Chain(relu))) == Chain(abs, sqrt, relu)
@test flatten_model(Chain(abs, sqrt, relu)) == Chain(abs, sqrt, relu)

# has_output_softmax
@test has_output_softmax(Chain(abs, sqrt, relu, softmax)) == true
@test has_output_softmax(Chain(abs, sqrt, relu, tanh)) == false
@test has_output_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == true
@test has_output_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) == false
@test has_output_softmax(Chain(Dense(5, 5, softmax), Dense(5, 5, softmax))) == true
@test has_output_softmax(Chain(Dense(5, 5, softmax), Dense(5, 5, relu))) == false
@test has_output_softmax(Chain(Dense(5, 5, softmax), Chain(Dense(5, 5, softmax)))) == true
@test has_output_softmax(Chain(Dense(5, 5, softmax), Chain(Dense(5, 5, relu)))) == false

# check_output_softmax
@test_throws ArgumentError check_output_softmax(Chain(abs, sqrt, relu, softmax))

# strip_softmax
d_softmax  = Dense(2, 2, softmax; init=pseudorand)
d_softmax2 = Dense(2, 2, softmax; init=pseudorand)
d_relu     = Dense(2, 2, relu; init=pseudorand)
d_identity = Dense(2, 2; init=pseudorand)
# flatten to remove softmax
m = strip_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax))))
@test m == Chain(Chain(abs), sqrt, Chain(Chain(identity)))
m1 = strip_softmax(Chain(d_relu, Chain(d_softmax)))
m2 = Chain(d_relu, Chain(d_identity))
x = rand(Float32, 2, 10)
@test typeof(m1) == typeof(m2)
@test m1(x) == m2(x)
# don't do anything if there is no softmax at the end
@test strip_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) ==
    Chain(Chain(abs), Chain(Chain(softmax)), sqrt)
@test strip_softmax(Chain(d_softmax, Chain(d_relu))) == Chain(d_softmax, Chain(d_relu))

# stabilize_denom
A = [1.0 0.0 1.0e-25; -1.0 -0.0 -1.0e-25]
S = @inferred stabilize_denom(A, 1e-3)
@test S ≈ [1.001 1e-3 1e-3; -1.001 1e-3 -1e-3]
S = @inferred stabilize_denom(Float32.(A), 1e-2)
@test S ≈ [1.01 1.0f-2 1.0f-2; -1.01 1.0f-2 -1.0f-2]

# batch_dim_view
A = [1 2; 3 4]
V = @inferred batch_dim_view(A)
@test size(V) == (2, 2, 1)
@test V[:, :, 1] == A

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
