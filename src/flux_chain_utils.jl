# To support `map` on Flux Chains containing both `Chain` and `Parallel` layers,
# we need a flexible, general purpose container, e.g. Tuple.
# We opt to use `ChainTuple` and `ParallelTuple` instead of `Chain` and `Parallel`
# to avoid type piracy.
for S in (:ChainTuple, :ParallelTuple)
    name = string(S)

    @eval begin
        """
            $($name)(xs)

        Thin wrapper around `Tuple` for use with Flux.jl models.

        Combining [`ChainTuple`](@ref) and [`ParallelTuple`](@ref),
        data `xs` can be stored while preserving the structure of a Flux model
        without risking type piracy.
        """
        struct ($S){T<:Tuple}
            vals::T
        end
        ($S)(xs...) = ($S)(xs)

        @forward $S.vals Base.getindex,
        Base.length,
        Base.first,
        Base.last,
        Base.iterate,
        Base.lastindex,
        Base.keys,
        Base.firstindex,
        Base.:(==)

        # Containers are equivalent if fields are equivalent
        Base.:(==)(a::$S, b::$S) = a.vals == b.vals

        # Print vals
        Base.show(io::IO, m::MIME"text/plain", t::$S) = print_vals(io, t)

        function print_vals(io::IO, t::$S, indent::Int=0)
            println(io, " "^indent, "$($name)(")
            for x in t
                print_vals(io, x, indent + 2)
            end
            println(io, " "^indent, ")", ifelse(indent != 0, ",", ""))
        end
    end
end
print_vals(io::IO, x, indent::Int=0) = println(io, " "^indent, x, ",")

"""
    chainmap(f, model)

`map` for Flux `Chains`. Applies the function `f` to all layers in a Flux model,
returning a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.
"""
function chainmap(f, x)
    if isleaf(x)
        return f(x)
    else
        T = constructor(x)
        vals = chainmap.(f, children(x))
        return T(vals...)
    end
end

# Implementation is strongly inspired by StructWalk.jl's postwalk
isleaf(c::Chain) = false
isleaf(p::Parallel) = false
isleaf(c::ChainTuple) = false
isleaf(p::ParallelTuple) = false
isleaf(x) = true

constructor(::Chain) = ChainTuple
constructor(::ChainTuple) = ChainTuple
constructor(::Parallel) = ParallelTuple
constructor(::ParallelTuple) = ParallelTuple

children(c::Chain) = c.layers
children(p::Parallel) = p.layers
children(c::ChainTuple) = c.vals
children(p::ParallelTuple) = p.vals
    heat_tail(xs)

Split input into head and tail.

## Examples
```julia-repl
julia> head_tail(1, 2, 3, 4)
(1, (2, 3, 4))

julia> head_tail((1, 2, 3, 4))
(1, (2, 3, 4))

julia> head_tail([1, 2, 3, 4])
(1, (2, 3, 4))

julia> head_tail(1, (2, 3), 4)
(1, ((2, 3), 4))

julia> head_tail(1)
(1, ())

julia> head_tail()
()
```
"""
head_tail(h, t...) = h, t
head_tail(h, t) = h, t
head_tail() = ()
head_tail(xs::Tuple) = head_tail(xs...)
head_tail(xs::AbstractVector) = head_tail(xs...)
head_tail(xs::Chain) = head_tail(xs...)

"""
    collect_activations(model, x)

Accumulates all hidden-layer and ouput activations of a Flux model,
returning a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

## Keyword arguments
- `collect_input`: Prepend the input `x` to the activations. Defaults to `true`.
"""
function collect_activations(model, x; collect_input=true)
    acts = _acts(model, x)
    collect_input && return x, acts
    return acts
end

# Split layer-tuples and Chains into head and tail
_acts(layers::Union{Tuple,AbstractVector}, x) = _acts(head_tail(layers)..., x)
_acts(c::Chain, x) = ChainTuple(_acts(c.layers, x))
# Parallel layers apply the functions above to each "branch"
_acts(p::Parallel, x) = ParallelTuple(nothing, [_acts(l, x) for l in p.layers]...)
# Special case: empty input tuple at the end of the recursion
_acts(::Tuple{}, x) = ()
# If none of the previous dispatches applied, we assume the layer is callable
_acts(layer, x) = layer(x)

function _acts(head, tail, x)
    aₕ = _acts(head, x)
    out = _output_activation(head, aₕ)
    aₜ = _acts(tail, out)

    # Splat regular tuples but not Chain-/ParallelTuple and arrays
    isa(aₜ, Tuple{}) && return aₕ
    isa(aₜ, Tuple) && return (aₕ, aₜ...)
    return (aₕ, aₜ)
end

_output_activation(layer, as) = __output_act(as)
function _output_activation(p::Parallel, as::ParallelTuple)
    outs = [_output_activation(l, a) for (l, a) in zip(p.layers, as.vals)]
    return p.connection(outs...)
end
__output_act(a) = a
__output_act(as::ChainTuple) = last(as)

"""
    last_element(model)
    last_element(chain_tuple)

Returns last layer of a Flux `Chain` or `ChainTuple`.
"""
last_element(c::Union{Chain,ChainTuple}) = last_element(c[end])
last_element(layer) = layer

"""
  check_output_softmax(model)

Check whether model has softmax activation on output.
Return the model if it doesn't, throw error otherwise.
"""
function check_output_softmax(model::Chain)
    if has_output_softmax(model)
        throw(ArgumentError("""Model contains softmax activation function on output.
        Call `strip_softmax` on your model."""))
    end
    return model
end

has_output_softmax(model::Chain) = has_output_softmax(last_element(model))
has_output_softmax(x) = is_softmax(x) || is_softmax(activation_fn(x))
is_softmax(x) = x isa SoftmaxActivation

"""
    strip_softmax(model)
    strip_softmax(layer)

Remove softmax activation on layer or model if it exists.
"""
strip_softmax(l) = copy_layer(l, l.weight, l.bias; σ=identity)
strip_softmax(::SoftmaxActivation) = identity

function strip_softmax(model::Chain)
    output_layer = last_element(model)
    !has_output_softmax(output_layer) && return model

    function _strip_softmax(layer)
        layer != output_layer && return layer
        return strip_softmax(layer)
    end
    _strip_softmax(c::Chain) = Chain(_strip_softmax.(c.layers)...)
    _strip_softmax(p::Parallel) = p # p.connection can't be softmax
    return _strip_softmax(model)
end
