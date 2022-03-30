using Flux
using ExplainableAI: flatten_model, has_output_softmax, check_output_softmax
using ExplainableAI: stabilize_denom, batch_dim_view, drop_batch_index

# flatten_model
@test flatten_model(Chain(Chain(Chain(abs)), sqrt, Chain(relu))) == Chain(abs, sqrt, relu)
@test flatten_model(Chain(abs, sqrt, relu)) == Chain(abs, sqrt, relu)
@test_deprecated flatten_chain(Chain(identity))

# has_output_softmax
@test has_output_softmax(Chain(abs, sqrt, relu, softmax)) == true
@test has_output_softmax(Chain(abs, sqrt, relu, tanh)) == false
@test has_output_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == true
@test has_output_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) == false

# check_output_softmax
@test_throws ArgumentError check_output_softmax(Chain(abs, sqrt, relu, softmax))

# strip_softmax
@test strip_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == Chain(abs, sqrt) # flatten to remove softmax
@test strip_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) ==
    Chain(Chain(abs), Chain(Chain(softmax)), sqrt) # don't do anything if there is no softmax at the end

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
