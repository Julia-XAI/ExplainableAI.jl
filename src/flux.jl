"""
activation(layer)

Return activation function of the layer.
In case the layer is unknown or no activation function is found, `nothing` is returned.
"""
activation(l::Dense) = l.σ
activation(l::Conv) = l.σ
activation(l::BatchNorm) = l.λ
activation(layer) = nothing # default for all other layer types

"""
    flatten_model(c)

Flatten a Flux chain containing Flux chains.
"""
function flatten_model(chain::Chain)
    if any(isa.(chain.layers, Chain))
        flatchain = Chain(vcat(_flatten_model.(chain.layers)...)...)
        return flatten_model(flatchain)
    end
    return chain
end
@deprecate flatten_chain(c) flatten_model(c)

_flatten_model(x) = x
_flatten_model(c::Chain) = [c.layers...]

is_softmax(x) = x isa SoftmaxActivation
has_output_softmax(x) = is_softmax(x) || is_softmax(activation(x))
has_output_softmax(model::Chain) = has_output_softmax(model[end])

"""
    check_output_softmax(model)

Check whether model has softmax activation on output.
Return the model if it doesn't, throw error otherwise.
"""
function check_output_softmax(model::Chain)
    if has_output_softmax(model)
        throw(ArgumentError("""Model contains softmax activation on output.
                            Call `strip_softmax` on your model first."""))
    end
    return model
end

"""
    strip_softmax(model)

Remove softmax activation on model output if it exists.
"""
function strip_softmax(model::Chain)
    if has_output_softmax(model)
        model = flatten_model(model)
        if is_softmax(model[end])
            return Chain(model.layers[1:(end - 1)]...)
        end
        return Chain(model.layers[1:(end - 1)]..., strip_softmax(model[end]))
    end
    return model
end
strip_softmax(l::Dense) = Dense(l.weight, l.bias, identity)
function strip_softmax(l::Conv)
    return Conv(identity, l.weight, l.bias, l.stride, l.pad, l.dilation, l.groups)
end

has_weight_and_bias(layer) = hasproperty(layer, :weight) && hasproperty(layer, :bias)
