## Group layers by type:
const ConvLayers = Union{Conv,DepthwiseConv,ConvTranspose,CrossCor}
const DropoutLayers = Union{Dropout,typeof(Flux.dropout),AlphaDropout}
const ReshapingLayers = Union{typeof(Flux.flatten)}
# Pooling layers
const MaxPoolLayers = Union{MaxPool,AdaptiveMaxPool,GlobalMaxPool}
const MeanPoolLayers = Union{MeanPool,AdaptiveMeanPool,GlobalMeanPool}
const PoolingLayers = Union{MaxPoolLayers,MeanPoolLayers}

_flatten_chain(x) = x
_flatten_chain(c::Chain) = [c.layers...]
"""
    flatten_chain(c)

Flatten a Flux chain containing Flux chains.
"""
function flatten_chain(chain::Chain)
    if any(isa.(chain.layers, Chain))
        flatchain = Chain(vcat(_flatten_chain.(chain.layers)...)...)
        return flatten_chain(flatchain)
    end
    return chain
end

is_softmax(layer) = layer isa Union{typeof(softmax),typeof(softmax!)}
has_output_softmax(x) = is_softmax(x)
has_output_softmax(model::Chain) = has_output_softmax(model[end])
"""
    check_ouput_softmax(model)

Check whether model has softmax activation on output.
Return the model if it doesn't, throw error otherwise.
"""
function check_ouput_softmax(model::Chain)
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
        model = flatten_chain(model)
        return Chain(model.layers[1:(end - 1)]...)
    end
    return model
end

# helper function to work around Flux.Zeros
function get_weights(layer)
    W = layer.weight
    b = layer.bias
    if typeof(b) <: Flux.Zeros
        b = zeros(eltype(W), size(W, 1))
    end
    return W, b
end

"""
    set_weights(layer, W, b)

Duplicate layer using weights W, b.
"""
set_weights(l::Conv, W, b) = Conv(l.σ, W, b, l.stride, l.pad, l.dilation, l.groups)
set_weights(l::Dense, W, b) = Dense(W, b, l.σ)
