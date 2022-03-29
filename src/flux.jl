## Group layers by type:
const ConvLayer = Union{Conv} # TODO: DepthwiseConv, ConvTranspose, CrossCor
const DropoutLayer = Union{Dropout,typeof(Flux.dropout),AlphaDropout}
const ReshapingLayer = Union{typeof(Flux.flatten)}
# Pooling layers
const MaxPoolLayer = Union{MaxPool,AdaptiveMaxPool,GlobalMaxPool}
const MeanPoolLayer = Union{MeanPool,AdaptiveMeanPool,GlobalMeanPool}
const PoolingLayer = Union{MaxPoolLayer,MeanPoolLayer}
# Activation functions that are similar to ReLU
const ReluLikeActivation = Union{
    typeof(relu),typeof(gelu),typeof(swish),typeof(softplus),typeof(mish)
}
# Layers & activation functions supported by LRP
const LRPSupportedLayer = Union{Dense,ConvLayer,DropoutLayer,ReshapingLayer,PoolingLayer}
const LRPSupportedActivation = Union{typeof(identity),ReluLikeActivation}

_flatten_model(x) = x
_flatten_model(c::Chain) = [c.layers...]
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

is_softmax(layer) = layer isa Union{typeof(softmax),typeof(softmax!)}
has_output_softmax(x) = is_softmax(x)
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
        return Chain(model.layers[1:(end - 1)]...)
    end
    return model
end

# helper function to work around Flux.Zeros
function get_params(layer)
    W = layer.weight
    b = layer.bias
    if typeof(b) <: Flux.Zeros
        b = zeros(eltype(W), size(W, 1))
    end
    return W, b
end

"""
    set_params(layer, W, b)

Duplicate layer using weights W, b.
"""
set_params(l::Conv, W, b) = Conv(l.σ, W, b, l.stride, l.pad, l.dilation, l.groups)
set_params(l::Dense, W, b) = Dense(W, b, l.σ)
