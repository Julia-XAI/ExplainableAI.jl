#===========#
# Composite #
#===========#

# A Composite is a container of primitives, which are sequentially applied

struct Composite{T<:Union{Tuple,AbstractVector}}
    primitives::T
end
Composite(primitives...) = Composite(primitives)
Composite(rule::AbstractLRPRule, prims...) = Composite((GlobalRule(rule), prims...))

(c::Composite)(model) = lrp_rules(model, c) # defined at end of file

# TODO: Documentation and new lrp_rules function, add to Changelog
# - new lrp_rules function
# - deprecation of LastNTypeRule
# - document new primitive interface

# TODO: rename primitives to e.g. "assigners" avoid confusion with LRP rules
# TODO: add LambdaRule
# TODO: add LambdaTypeRule

#=================#
# Rule primitives #
#=================#

# All primitives need to implement the following interfaces: # TODO

abstract type AbstractCompositePrimitive end
abstract type AbstractRulePrimitive <: AbstractCompositePrimitive end

"""
    GlobalRule(rule)

Composite primitive that applies LRP-rule `rule` to all layers in the model.

See [`Composite`](@ref) for an example.
"""
struct GlobalRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end

"""
    LayerRule(n, rule)

Composite primitive that applies LRP-rule `rule` to the `n`-th layer in the model.

See [`Composite`](@ref) for an example.
"""
struct LayerRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    n::Int
    rule::R
end

"""
    RangeRule(range, rule)

Composite primitive that applies LRP-rule `rule` to the specified positional `range`
of layers in the model.

See [`Composite`](@ref) for an example.
"""
struct RangeRule{T<:AbstractRange,R<:AbstractLRPRule} <: AbstractRulePrimitive
    range::T
    rule::R
end

"""
    FirstLayerRule(rule)

Composite primitive that applies LRP-rule `rule` to the first layer in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end

"""
    LastLayerRule(rule)

Composite primitive that applies LRP-rule `rule` to the last layer in the model.

See [`Composite`](@ref) for an example.
"""
struct LastLayerRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end

#=====================#
# TypeRule primitives #
#=====================#

abstract type AbstractTypeRulePrimitive <: AbstractCompositePrimitive end
const TypeRulePair = Pair{<:Type,<:AbstractLRPRule}

"""
    GlobalTypeRule(map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct GlobalTypeRule{T<:AbstractVector{<:TypeRulePair}} <: AbstractTypeRulePrimitive
    map::T
end

"""
    RangeTypeRule(range, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the specified `range` of layers in the model.

See [`Composite`](@ref) for an example.
"""
struct RangeTypeRule{R<:AbstractRange,T<:AbstractVector{<:TypeRulePair}} <:
       AbstractTypeRulePrimitive
    range::R
    map::T
end

"""
    FirstNTypeRule(n, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the first `n` layers in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstNTypeRule{T<:AbstractVector{<:TypeRulePair}} <: AbstractTypeRulePrimitive
    n::Int
    map::T
end

"""
    FirstLayerTypeRule(map)

Composite primitive that maps the type of the first layer of the model to LRP rules
based on a list of type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerTypeRule{T<:AbstractVector{<:TypeRulePair}} <: AbstractTypeRulePrimitive
    map::T
end

"""
    LastLayerTypeRule(map)

Composite primitive that maps the type of the last layer of the model to LRP rules
based on a list of type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct LastLayerTypeRule{T<:AbstractVector{<:TypeRulePair}} <: AbstractTypeRulePrimitive
    map::T
end

# Convenience constructors
GlobalTypeRule(ps::Vararg{TypeRulePair})     = GlobalTypeRule([ps...])
RangeTypeRule(r, ps::Vararg{TypeRulePair})   = RangeTypeRule(r, [ps...])
FirstLayerTypeRule(ps::Vararg{TypeRulePair}) = FirstLayerTypeRule([ps...])
LastLayerTypeRule(ps::Vararg{TypeRulePair})  = LastLayerTypeRule([ps...])
FirstNTypeRule(n, ps::Vararg{TypeRulePair})  = FirstNTypeRule(n, [ps...])

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

    get_rule(r::LayerRule, _, key) = ifelse(first(key) == r.n, r.rule, nothing)
    get_rule(r::GlobalRule, _, _key) = r.rule
    get_rule(r::RangeRule, _, key) = ifelse(first(key) ∈ r.range, r.rule, nothing)
    get_rule(r::FirstLayerRule, _, key) = ifelse(key == first_key, r.rule, nothing)
    get_rule(r::LastLayerRule, _, key) = ifelse(key == last_key, r.rule, nothing)

    function get_rule(r::GlobalTypeRule, layer, _key)
        return get_type_rule(layer, r.map)
    end
    function get_rule(r::RangeTypeRule, layer, key)
        return ifelse(first(key) ∈ r.range, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::FirstLayerTypeRule, layer, key)
        return ifelse(key == first_key, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::LastLayerTypeRule, layer, key)
        return ifelse(key == last_key, get_type_rule(layer, r.map), nothing)
    end
    function get_rule(r::FirstNTypeRule, layer, key)
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
* [`LayerRule`](@ref) to apply a rule to the `n`-th layer of a model
* [`GlobalRule`](@ref) to apply a rule to all layers
* [`RangeRule`](@ref) to apply a rule to a positional range of layers
* [`FirstLayerRule`](@ref) to apply a rule to the first layer
* [`LastLayerRule`](@ref) to apply a rule to the last layer

To apply a set of rules to layers based on their type, use:
* [`GlobalTypeRule`](@ref) to apply a dictionary that maps layer types to LRP-rules
* [`RangeTypeRule`](@ref) for a `TypeRule` on generalized ranges
* [`FirstLayerTypeRule`](@ref) for a `TypeRule` on the first layer of a model
* [`LastLayerTypeRule`](@ref) for a `TypeRule` on the last layer
* [`FirstNTypeRule`](@ref) for a `TypeRule` on the first `n` layers

# Example
Using a flattened VGG11 model:
```julia-repl
julia> composite = Composite(
           GlobalTypeRule(
               ConvLayer => AlphaBetaRule(),
               Dense => EpsilonRule(),
               PoolingLayer => EpsilonRule(),
               DropoutLayer => PassRule(),
               ReshapingLayer => PassRule(),
           ),
           FirstNTypeRule(7, Conv => FlatRule()),
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
