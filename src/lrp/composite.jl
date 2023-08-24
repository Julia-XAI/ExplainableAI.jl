#===========#
# Composite #
#===========#

# A Composite is a container of primitives, which are sequentially applied

struct Composite{T<:Union{Tuple,AbstractVector}}
    primitives::T
end
Composite(primitives...) = Composite(primitives)
Composite(rule::AbstractLRPRule, prims...) = Composite((GlobalMap(rule), prims...))

# TODO: add LambdaRule
# TODO: add LambdaTypeMap

#=================#
# Rule primitives #
#=================#

abstract type AbstractCompositePrimitive end
abstract type AbstractCompositeMap <: AbstractCompositePrimitive end

"""
    GlobalMap(rule)

Composite primitive that maps LRP-rule `rule` to all layers in the model.

See [`Composite`](@ref) for an example.
"""
struct GlobalMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

"""
    LayerMap(n, rule)

Composite primitive that maps LRP-rule `rule` to the `n`-th layer in the model.

See [`Composite`](@ref) for an example.
"""
struct LayerMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    n::Int
    rule::R
end

"""
    RangeMap(range, rule)

Composite primitive that maps LRP-rule `rule` to the specified positional `range`
of layers in the model.

See [`Composite`](@ref) for an example.
"""
struct RangeMap{T<:AbstractRange,R<:AbstractLRPRule} <: AbstractCompositeMap
    range::T
    rule::R
end

"""
    FirstLayerMap(rule)

Composite primitive that maps LRP-rule `rule` to the first layer in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

"""
    LastLayerMap(rule)

Composite primitive that maps LRP-rule `rule` to the last layer in the model.

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
FirstLayerTypeMap(ps::Vararg{TypeMapPair}) = FirstLayerTypeMap([ps...])
LastLayerTypeMap(ps::Vararg{TypeMapPair})  = LastLayerTypeMap([ps...])
FirstNTypeMap(n, ps::Vararg{TypeMapPair})  = FirstNTypeMap(n, [ps...])

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
    lrp_rules(model, composite)

Apply a composite to obtain LRP-rules for a given Flux model.
"""
function lrp_rules(model, c::Composite)
    keys = chainkeys(model)
    first_key = first_element(keys)
    last_key = last_element(keys)

    get_rule(r::LayerMap, _, key) = ifelse(first(key) == r.n, r.rule, nothing)
    get_rule(r::GlobalMap, _, _key) = r.rule
    get_rule(r::RangeMap, _, key) = ifelse(first(key) ∈ r.range, r.rule, nothing)
    get_rule(r::FirstLayerMap, _, key) = ifelse(key == first_key, r.rule, nothing)
    get_rule(r::LastLayerMap, _, key) = ifelse(key == last_key, r.rule, nothing)

    function get_rule(r::GlobalTypeMap, layer, _key)
        return get_type_rule(layer, r.map)
    end
    function get_rule(r::RangeTypeMap, layer, key)
        return ifelse(first(key) ∈ r.range, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::FirstLayerTypeMap, layer, key)
        return ifelse(key == first_key, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::LastLayerTypeMap, layer, key)
        return ifelse(key == last_key, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::FirstNTypeMap, layer, key)
        return ifelse(first(key) ∈ 1:(r.n), get_type_rule(layer, r.map), nothing)
    end

    # The last rule returned from a composite primitive is assigned to the layer.
    # This is implemented by returning the first rule in reverse order:
    function match_rule(layer, key)
        for primitive in reverse(c.primitives)
            rule = get_rule(primitive, layer, key)
            !isnothing(rule) && return rule
        end
        return ZeroRule() # else if no assignment was found, return default rule
    end
    return chainzip(match_rule, model, keys) # construct ChainTuple of rules
end

"""
    Composite([default_rule=LRPZero()], primitives...)

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
Using a flattened VGG11 model:
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

julia> analyzer = LRP(model, composite);

julia> analyzer.rules
19-element Vector{AbstractLRPRule}:
 FlatRule()
 EpsilonRule{Float32}(1.0f-6)
 FlatRule()
 EpsilonRule{Float32}(1.0f-6)
 FlatRule()
 FlatRule()
 EpsilonRule{Float32}(1.0f-6)
 AlphaBetaRule{Float32}(2.0f0, 1.0f0)
 AlphaBetaRule{Float32}(2.0f0, 1.0f0)
 EpsilonRule{Float32}(1.0f-6)
 AlphaBetaRule{Float32}(2.0f0, 1.0f0)
 AlphaBetaRule{Float32}(2.0f0, 1.0f0)
 EpsilonRule{Float32}(1.0f-6)
 PassRule()
 EpsilonRule{Float32}(1.0f-6)
 PassRule()
 EpsilonRule{Float32}(1.0f-6)
 PassRule()
 EpsilonRule{Float32}(1.0f-6)
```
"""
Composite
