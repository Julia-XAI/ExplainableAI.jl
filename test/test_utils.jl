using Flux
using ExplainabilityMethods: flatten_model, has_output_softmax, stabilize_denom

# flatten_model
@test flatten_model(Chain(Chain(Chain(abs)), sqrt, Chain(relu))) == Chain(abs, sqrt, relu)
@test flatten_model(Chain(abs, sqrt, relu)) == Chain(abs, sqrt, relu)
@test_deprecated flatten_chain(Chain(identity))

# has_output_softmax
@test has_output_softmax(Chain(abs, sqrt, relu, softmax)) == true
@test has_output_softmax(Chain(abs, sqrt, relu, tanh)) == false
@test has_output_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == true
@test has_output_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) == false

# strip_softmax
@test strip_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == Chain(abs, sqrt) # flatten to remove softmax
@test strip_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) ==
    Chain(Chain(abs), Chain(Chain(softmax)), sqrt) # don't do anything if there is no softmax at the end

# stabilize_denom
A = [1.0 0; -0 -1.0e-25]
@test stabilize_denom(A; eps=1e-3) ≈ [1.001 1e-3; 1e-3 -1e-3]
