using Flux
using ExplainabilityMethods: flatten_chain, has_output_softmax

@test flatten_chain(Chain(Chain(Chain(abs)), sqrt, Chain(relu))) == Chain(abs, sqrt, relu)
@test flatten_chain(Chain(abs, sqrt, relu)) == Chain(abs, sqrt, relu)

@test has_output_softmax(Chain(abs, sqrt, relu, softmax)) == true
@test has_output_softmax(Chain(abs, sqrt, relu, tanh)) == false
@test has_output_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == true
@test has_output_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) == false

@test strip_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == Chain(abs, sqrt) # flatten to remove softmax
@test strip_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) ==
    Chain(Chain(abs), Chain(Chain(softmax)), sqrt) # don't do anything if there is no softmax at the end
