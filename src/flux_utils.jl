# To support LRP on Flux Chains containing both `Chain` and `Parallel` layers,
# we need flexible, distinguishable, general purpose "containers".
# Two Tuple wrapper types are introduced: `ChainTuple` and `ParallelTuple`:
struct ChainTuple{T<:Tuple}
    vals::T
end
struct ParallelTuple{T<:Tuple}
    vals::T
end

ChainTuple(xs...) = ChainTuple(xs)
ParallelTuple(xs...) = ParallelTuple(xs)

# Foward Base functions to wrapped Tuple `vals`:
@forward ChainTuple.vals Base.getindex,
Base.length, Base.first, Base.last, Base.iterate, Base.lastindex, Base.keys,
Base.firstindex
@forward ParallelTuple.vals Base.getindex,
Base.length, Base.first, Base.last, Base.iterate, Base.lastindex, Base.keys,
Base.firstindex

"""
    collect_activations(model, input)

Collect all hidden layer activations of a Flux model.
The model can contain other `Chain` and `Parallel` layers.
"""
collect_activations(model, input) = (input, _collect_acts(model, input)...)

# Split head and tail
_head_tail(head, tail...) = head, tail
_head_tail(head, tail) = head, tail
_head_tail() = ()
_head_tail(xs::Tuple) = _head_tail(xs...)
_head_tail(xs::AbstractVector) = _head_tail(xs...)

_collect_acts(layers, x) = _collect_acts(_head_tail(layers)..., x)
_collect_acts(::Tuple{}, x) = ()

function _collect_acts(head, tail, x)
    ret, out = _collect_acts_head(head, x)
    coll_tail = _collect_acts(tail, out)
    isa(coll_tail, Tuple) && return (ret, coll_tail...) # don't splat Chain-/ParallelTuple
    return (ret, coll_tail)
end
function _collect_acts_head(layer, x)
    y = layer(x)
    return y, y # ret, out
end
function _collect_acts_head(c::Chain, x)
    t = ChainTuple(_collect_acts(c.layers, x)...)
    return t, last(t) # ret, out
end

"""
    activation(layer)

Return activation function of the layer.
In case the layer is unknown or no activation function is found, `nothing` is returned.
"""
activation(layer) = nothing
activation(l::Dense)         = l.σ
activation(l::Conv)          = l.σ
activation(l::CrossCor)      = l.σ
activation(l::ConvTranspose) = l.σ
activation(l::BatchNorm)     = l.λ

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
strip_softmax(l) = copy_layer(l, l.weight, l.bias; σ=identity)

has_weight(layer) = hasproperty(layer, :weight)
has_bias(layer) = hasproperty(layer, :bias)
has_weight_and_bias(layer) = has_weight(layer) && has_bias(layer)

"""
    copy_layer(layer, W, b, [σ=identity])

Copy layer using weights `W` and `b`. The activation function `σ` can also be set,
defaulting to `identity`.
"""
copy_layer(::Dense, W, b; σ=identity) = Dense(W, b, σ)
copy_layer(l::Conv, W, b; σ=identity) = Conv(σ, W, b, l.stride, l.pad, l.dilation, l.groups)
function copy_layer(l::ConvTranspose, W, b; σ=identity)
    return ConvTranspose(σ, W, b, l.stride, l.pad, l.dilation, l.groups)
end
function copy_layer(l::CrossCor, W, b; σ=identity)
    return CrossCor(σ, W, b, l.stride, l.pad, l.dilation)
end
