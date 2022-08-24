"""
activation(layer)

Return activation function of the layer.
In case the layer is unknown or no activation function is found, `nothing` is returned.
"""
activation(l::Dense) = l.σ
activation(l::Conv) = l.σ
activation(l::ConvTranspose) = l.σ
activation(l::CrossCor) = l.σ
activation(l::BatchNorm) = l.λ
activation(layer) = nothing # default for all other layer types

function has_activation(layer)
    hasproperty(layer, :σ) && return true
    hasproperty(layer, :λ) && return true
    return !isnothing(activation(layer))
end

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
function require_weight_and_bias(rule, layer)
    !has_weight_and_bias(layer) && throw(
        ArgumentError(
            "$rule requires linear layer with weight and bias parameters, got $layer."
        ),
    )
    return nothing
end

# LRP requires computing so called pre-activations `z`.
# These correspond to calling a layer without applying its activation function.
preactivation(layer) = x -> preactivation(layer, x)
function preactivation(d::Dense, x::AbstractVecOrMat)
    return d.weight * x .+ d.bias
end
function preactivation(d::Dense, x::AbstractArray)
    return reshape(d(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end
function preactivation(c::Conv, x)
    cdims = Flux.DenseConvDims(
        x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation, groups=c.groups
    )
    return Flux.conv(x, c.weight, cdims) .+ Flux.conv_reshape_bias(c)
end

function preactivation(c::ConvTranspose, x)
    cdims = Flux.conv_transpose_dims(c, x)
    return Flux.∇conv_data(x, c.weight, cdims) .+ Flux.conv_reshape_bias(c)
end
function preactivation(c::CrossCor, x)
    cdims = Flux.DenseConvDims(
        x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation
    )
    return Flux.crosscor(x, c.weight, cdims) .+ Flux.conv_reshape_bias(c)
end
function preactivation(l, x)
    has_activation(l) &&
        error("""Layer $l contains an activation function and therefore requires an
            implementation of `preactivation(layer, input)`""")
    return l(x)
end
