#============================#
# ChainTuple & ParallelTuple #
#============================#

# To support map and zip on Flux Chains containing both `Chain` and `Parallel` layers,
# we need a flexible, general purpose container, e.g. a Tuple.
# We opt to introduce `ChainTuple` and `ParallelTuple` instead of `Chain` and `Parallel`
# to avoid type piracy.

"""
    ChainTuple(xs)

Thin wrapper around `Tuple` for use with Flux.jl models.

Combining [`ChainTuple`](@ref) and [`ParallelTuple`](@ref),
data `xs` can be stored while preserving the structure of a Flux model
without risking type piracy.
"""
struct ChainTuple{T<:Tuple}
    vals::T
end

"""
    ParallelTuple(xs)

Thin wrapper around `Tuple` for use with Flux.jl models.

Combining [`ChainTuple`](@ref) and [`ParallelTuple`](@ref),
data `xs` can be stored while preserving the structure of a Flux model
without risking type piracy.
"""
struct ParallelTuple{T<:Tuple}
    vals::T
end

for T in (:ChainTuple, :ParallelTuple)
    name = string(T)

    @eval begin
        ($T)(xs...) = ($T)(xs)

        @forward $T.vals Base.getindex,
        Base.length,
        Base.first,
        Base.last,
        Base.iterate,
        Base.lastindex,
        Base.keys,
        Base.firstindex,
        Base.:(==)
        Base.similar

        # Containers are equivalent if fields are equivalent
        Base.:(==)(a::$T, b::$T) = a.vals == b.vals

        # Print vals
        Base.show(io::IO, m::MIME"text/plain", t::$T) = print_vals(io, t)

        function print_vals(io::IO, t::$T, indent::Int=0)
            println(io, " "^indent, $name, "(")
            for x in t
                print_vals(io, x, indent + 2)
            end
            println(io, " "^indent, ")", ifelse(indent != 0, ",", ""))
        end
    end # eval
end
print_vals(io::IO, x, indent::Int=0) = println(io, " "^indent, x, ",")

#=====================#
# chainmap & chainzip #
#=====================#

# The following implementation of map and zip on Chains and Parallel layers
# are strongly inspired by StructWalk.jl's postwalk function.

isleaf(c::Chain)         = false
isleaf(p::Parallel)      = false
isleaf(c::ChainTuple)    = false
isleaf(p::ParallelTuple) = false
isleaf(x)                = true

constructor(::Chain)         = ChainTuple
constructor(::ChainTuple)    = ChainTuple
constructor(::Parallel)      = ParallelTuple
constructor(::ParallelTuple) = ParallelTuple

children(c::Chain)         = c.layers
children(p::Parallel)      = p.layers
children(c::ChainTuple)    = c.vals
children(p::ParallelTuple) = p.vals

"""
    chainmap(f, x)

`map` for Flux models. Applies the function `f` to nested structures of `Chain`s
and `Parallel` layers.
Returns a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.

See also [`chainzip`](@ref).
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

"""
    chainall(f, model)

Determines whether `f` returns `true` for all elements of a Flux `Chain` `x`.
Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.
"""
function chainall(f, x)
    isleaf(x) && return f(x)
    return all(chainall.(f, children(x)))
end

"""
:q
Sequentially enumerate all layers in a Flux model.
Nested `Chain` and `Parallel` layers will result in tuples of indices.

# Example:
```julia-repl
julia> d = Dense(2, 2);

julia> model = Chain(d, Parallel(+, d, d, Chain(d, d)), d);

julia> chainindices(model)
ChainTuple(
  (1,),
  ParallelTuple(
    (2, 1),
    (2, 2),
    ChainTuple(
      (2, 3, 1),
      (2, 3, 2),
    ),
  ),
  (3,),
)
```
"""
chainindices(model) = chainindices(model, tuple())
function chainindices(x, key)
    if isleaf(x)
        return key
    else
        T = constructor(x)
        keys = map(i -> (key..., i), 1:length(children(x)))
        return T(chainindices.(children(x), keys)...)
    end
end

"""
    show_layer_indices(model)

Print layer indices of Flux models.
This is primarily a utility to help define [`LayerMap`](@ref) primitives.
"""
show_layer_indices(model) = chainindices(model)

"""
    chainzip(f, x, y)
    chainzip(f, xs...)

`zip` for Flux models. Applies the function `f` to nested structures of `Chain`s
and `Parallel` layers. Assumes that arguments have the same structure.
Returns a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.

See also [`chainmap`](@ref).
"""
function chainzip(f, xs...)
    if all(isleaf, xs)
        return f(xs...)
    else
        constructors = constructor.(xs)
        T = first(constructors)
        # Assume that arguments xs are zippable if constructors match:
        all(c -> c == T, constructors) || error("Cannot chainzip arguments $xs.")
        vals = chainzip.(f, children.(xs)...)
        return T(vals...)
    end
end

#===============#
# Flatten model #
#===============#

"""
    flatten_model(model)

Flatten a Flux `Chain` containing `Chain`s.
"""
flatten_model(x) = chainflatten(x)

"""
    chainflatten(chain)

Flatten a Flux `Chain` containing `Chain`s. Also works with `ChainTuple`s.
"""
function chainflatten(c::Chain)
    if length(c.layers) == 1
        return Chain(_chainflatten(c))
    else
        return Chain(_chainflatten(c)...)
    end
end
function chainflatten(c::ChainTuple)
    if length(c.vals) == 1
        return ChainTuple(_chainflatten(c))
    else
        return ChainTuple(_chainflatten(c)...)
    end
end
_chainflatten(c::Chain)      = mapreduce(_chainflatten, vcat, c.layers)
_chainflatten(c::ChainTuple) = mapreduce(_chainflatten, vcat, c.vals)

chainflatten(p::Parallel)       = _chainflatten(p)
chainflatten(p::ParallelTuple)  = _chainflatten(p)
_chainflatten(p::Parallel)      = Parallel(p.connection, chainflatten.(p.layers))
_chainflatten(p::ParallelTuple) = ParallelTuple(chainflatten.(p.vals))

chainflatten(x) = x
_chainflatten(x) = x

#=========================#
# Strip output activation #
#=========================#
"""
    first_element(model)

Returns last layer of a Flux `Chain` or `ChainTuple`.
"""
first_element(c::Union{Chain,ChainTuple}) = first_element(c[1])
first_element(layer) = layer

"""
    last_element(model)

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
