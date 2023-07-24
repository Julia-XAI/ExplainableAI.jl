# To support `map` on Flux Chains containing both `Chain` and `Parallel` layers,
# we need a flexible, general purpose container, e.g. Tuple.
# We opt to use `ChainTuple` and `ParallelTuple` instead of `Chain` and `Parallel`
# to avoid type piracy.
"""
    ChainTuple(xs)

Thin wrapper around `Tuple` for use with Flux.jl models.

Together with [`ParallelTuple`](@ref), this can be used to store data `xs`
while preserving the structure of a Flux model without risking type piracy.

See also [`ParallelTuple`](@ref), [`chainmap`](@ref).
"""
struct ChainTuple{T<:Tuple}
    vals::T
end
ChainTuple(xs...) = ChainTuple(xs)

@forward ChainTuple.vals Base.getindex,
Base.length,
Base.first,
Base.last,
Base.iterate,
Base.lastindex,
Base.keys,
Base.firstindex,
Base.:(==)

Base.:(==)(a::ChainTuple, b::ChainTuple) = a.vals == b.vals

"""
    ParallelTuple(connection, xs)

Thin wrapper around `Tuple` for use with Flux.jl models.

Together with [`ChainTuple`](@ref), this can be used to store data `xs`
while preserving the structure of a Flux model without risking type piracy.

See also [`ChainTuple`](@ref), [`chainmap`](@ref).
"""
struct ParallelTuple{C,T<:Tuple}
    connection::C
    vals::T
end
ParallelTuple(connection, xs...) = ParallelTuple(connection, xs)

@forward ParallelTuple.vals Base.getindex,
Base.length,
Base.first,
Base.last,
Base.iterate,
Base.lastindex,
Base.keys,
Base.firstindex,
Base.:(==)

function Base.:(==)(a::ParallelTuple, b::ParallelTuple)
    return a.connection == b.connection && a.vals == b.vals
end

Base.show(io::IO, m::MIME"text/plain", t::ChainTuple) = _show_tuple(io, t, 0)
Base.show(io::IO, m::MIME"text/plain", t::ParallelTuple) = _show_tuple(io, t, 0)

function _show_tuple(io::IO, ct::ChainTuple, indent::Int)
    println(io, " "^indent, "ChainTuple(")
    for x in ct
        _show_tuple(io, x, indent + 2)
    end
    println(io, " "^indent, ")", ifelse(indent != 0, ",", ""))
end
function _show_tuple(io::IO, pt::ParallelTuple, indent::Int)
    println(io, " "^indent, "ParallelTuple(")
    print(io, " "^(indent + 2), pt.connection, ", ")
    printstyled(io, "# connection \n"; color=:light_black)
    for x in pt
        _show_tuple(io, x, indent + 2)
    end
    println(io, " "^indent, ")", ifelse(indent != 0, ",", ""))
end
_show_tuple(io::IO, layer, indent::Int) = println(io, " "^indent, layer, ",")

"""
    chainmap(f, model)
    chainmap(f, g, model)

`map` for Flux `Chains`. Applies the function `f` to all layers in a Flux model,
returning a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

## Optional arguments
A second function `g(p::Parallel) = connection` can be passed,
which takes a `Parallel` layer `p` as input and sets the `connection` field
of the constructed `ParallelTuple`.
If no function is specified, `connection` will be set to `nothing`.
"""
chainmap(f, layer) = chainmap(f, Returns(nothing), layer)

chainmap(f, g, c::Chain) = ChainTuple(chainmap.(f, g, c.layers)...)
chainmap(f, g, p::Parallel) = ParallelTuple(g(p), chainmap.(f, g, p.layers)...)
chainmap(f, _g, layer) = f(layer)

# chainmap can be re-applied on results:
chainmap(f, g, c::ChainTuple) = ChainTuple(chainmap.(f, g, c.vals)...)
chainmap(f, g, p::ParallelTuple) = ChainTuple(chainmap.(f, g, p.vals)...)

"""
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
