#===========#
# Composite #
#===========#

# A Composite is a container of primitives, which are sequentially applied

struct Composite{T<:Union{Tuple,AbstractVector}}
    primitives::T
end
Composite(primitives...) = Composite(primitives)
Composite(rule::AbstractLRPRule, prims...) = Composite((GlobalMap(rule), prims...))

#=================#
# Rule primitives #
#=================#

abstract type AbstractCompositePrimitive end
abstract type AbstractCompositeMap <: AbstractCompositePrimitive end

"""
    GlobalMap(rule)

Composite primitive that maps an LRP-rule to all layers in the model.

See [`Composite`](@ref) for an example.
"""
struct GlobalMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

"""
    LayerMap(index, rule)

Composite primitive that maps an LRP-rule to all layers in the model at the given index.
The index can either be an integer or a tuple of integers to map a rule to a specific layer
in nested Flux `Chain`s.

See [`show_layer_indices`](@ref) to print layer indices and [`Composite`](@ref) for an example.
"""
struct LayerMap{I<:Union{Integer,Tuple},R<:AbstractLRPRule} <: AbstractCompositeMap
    index::I
    rule::R
end

"""
    RangeMap(range, rule)

Composite primitive that maps an LRP-rule to the specified positional `range`
of layers in the model.

See [`Composite`](@ref) for an example.
"""
struct RangeMap{T<:AbstractRange,R<:AbstractLRPRule} <: AbstractCompositeMap
    range::T
    rule::R
end

"""
    FirstLayerMap(rule)

Composite primitive that maps an LRP-rule to the first layer in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

"""
    LastLayerMap(rule)

Composite primitive that maps an LRP-rule to the last layer in the model.

See [`Composite`](@ref) for an example.
"""
struct LastLayerMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

#=====================#
# TypeMap primitives #
#=====================#

abstract type AbstractCompositeTypeMap <: AbstractCompositePrimitive end
const TypeMapPair = Pair{<:Type,<:AbstractLRPRule}

"""
    GlobalTypeMap(map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct GlobalTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    map::T
end

"""
    RangeTypeMap(range, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the specified `range` of layers in the model.

See [`Composite`](@ref) for an example.
"""
struct RangeTypeMap{R<:AbstractRange,T<:AbstractVector{<:TypeMapPair}} <:
       AbstractCompositeTypeMap
    range::R
    map::T
end

"""
    FirstNTypeMap(n, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the first `n` layers in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstNTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    n::Int
    map::T
end

"""
    FirstLayerTypeMap(map)

Composite primitive that maps the type of the first layer of the model to LRP rules
based on a list of type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    map::T
end

"""
    LastLayerTypeMap(map)

Composite primitive that maps the type of the last layer of the model to LRP rules
based on a list of type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct LastLayerTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    map::T
end

# Convenience constructors
GlobalTypeMap(ps::Vararg{TypeMapPair})     = GlobalTypeMap([ps...])
RangeTypeMap(r, ps::Vararg{TypeMapPair})   = RangeTypeMap(r, [ps...])
FirstNTypeMap(n, ps::Vararg{TypeMapPair})  = FirstNTypeMap(n, [ps...])
FirstLayerTypeMap(ps::Vararg{TypeMapPair}) = FirstLayerTypeMap([ps...])
LastLayerTypeMap(ps::Vararg{TypeMapPair})  = LastLayerTypeMap([ps...])

#=====================#
# LRP-rule assignment #
#=====================#

function get_type_rule(layer, map)
    for (T, rule) in map
        if layer isa T
            return rule
        end
    end
    return nothing
end

"""
    in_branch(a, b)

Viewing index tuples `a` and `b` as positions on a tree-like data structure,
this checks whether `a` is on the same "branch" as `b`.

## Examples
```julia-repl
julia> in_branch((1, 2), 1)
true

julia> in_branch((1, 2), 2)
false

julia> in_branch((1, 2), (1, 2))
true

julia> in_branch((1, 2, 3), (1, 2))
true

julia> in_branch((1, 2), (1, 2, 3))
false
```
"""
in_branch(a::Integer, b::Integer) = a == b
in_branch(a::Tuple, b::Integer) = first(a) == b
function in_branch(a::Tuple, b::Tuple)
    length(a) < length(b) && return false
    for i in eachindex(b)
        a[i] != b[i] && return false
    end
    return true
end

"""
    lrp_rules(model, composite)

Apply a composite to obtain LRP-rules for a given Flux model.
"""
function lrp_rules(model, c::Composite)
    indices = chainindices(model)
    idx_first = first_element(indices)
    idx_last = last_element(indices)

    get_rule(r::LayerMap, _, idx) = ifelse(in_branch(idx, r.index), r.rule, nothing)
    get_rule(r::GlobalMap, _, _idx) = r.rule
    get_rule(r::RangeMap, _, idx) = ifelse(first(idx) ∈ r.range, r.rule, nothing)
    get_rule(r::FirstLayerMap, _, idx) = ifelse(idx == idx_first, r.rule, nothing)
    get_rule(r::LastLayerMap, _, idx) = ifelse(idx == idx_last, r.rule, nothing)

    function get_rule(r::GlobalTypeMap, layer, _idx)
        return get_type_rule(layer, r.map)
    end
    function get_rule(r::RangeTypeMap, layer, idx)
        return ifelse(first(idx) ∈ r.range, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::FirstLayerTypeMap, layer, idx)
        return ifelse(idx == idx_first, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::LastLayerTypeMap, layer, idx)
        return ifelse(idx == idx_last, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::FirstNTypeMap, layer, idx)
        return ifelse(first(idx) ∈ 1:(r.n), get_type_rule(layer, r.map), nothing)
    end

    # The last rule returned from a composite primitive is assigned to the layer.
    # This is implemented by returning the first rule in reverse order:
    function match_rule(layer, idx)
        for primitive in reverse(c.primitives)
            rule = get_rule(primitive, layer, idx)
            !isnothing(rule) && return rule
        end
        return ZeroRule() # else if no assignment was found, return default rule
    end
    return chainzip(match_rule, model, indices) # construct ChainTuple of rules
end

"""
    Composite(primitives...)
    Composite(default_rule, primitives...)

Automatically contructs a list of LRP-rules by sequentially applying composite primitives.

# Primitives
To apply a single rule, use:
* [`LayerMap`](@ref) to apply a rule to the `n`-th layer of a model
* [`GlobalMap`](@ref) to apply a rule to all layers
* [`RangeMap`](@ref) to apply a rule to a positional range of layers
* [`FirstLayerMap`](@ref) to apply a rule to the first layer
* [`LastLayerMap`](@ref) to apply a rule to the last layer

To apply a set of rules to layers based on their type, use:
* [`GlobalTypeMap`](@ref) to apply a dictionary that maps layer types to LRP-rules
* [`RangeTypeMap`](@ref) for a `TypeMap` on generalized ranges
* [`FirstLayerTypeMap`](@ref) for a `TypeMap` on the first layer of a model
* [`LastLayerTypeMap`](@ref) for a `TypeMap` on the last layer
* [`FirstNTypeMap`](@ref) for a `TypeMap` on the first `n` layers

# Example
Using a VGG11 model:
```julia-repl
julia> composite = Composite(
           GlobalTypeMap(
               ConvLayer => AlphaBetaRule(),
               Dense => EpsilonRule(),
               PoolingLayer => EpsilonRule(),
               DropoutLayer => PassRule(),
               ReshapingLayer => PassRule(),
           ),
           FirstNTypeMap(7, Conv => FlatRule()),
       );

julia> analyzer = LRP(model, composite)
LRP(
  Conv((3, 3), 3 => 64, relu, pad=1)    => FlatRule(),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 64 => 128, relu, pad=1)  => FlatRule(),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 128 => 256, relu, pad=1) => FlatRule(),
  Conv((3, 3), 256 => 256, relu, pad=1) => FlatRule(),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 256 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  Conv((3, 3), 512 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 512 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  Conv((3, 3), 512 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  MLUtils.flatten                       => PassRule(),
  Dense(25088 => 4096, relu)            => EpsilonRule{Float32}(1.0f-6),
  Dropout(0.5)                          => PassRule(),
  Dense(4096 => 4096, relu)             => EpsilonRule{Float32}(1.0f-6),
  Dropout(0.5)                          => PassRule(),
  Dense(4096 => 1000)                   => EpsilonRule{Float32}(1.0f-6),
)
```
"""
Composite
