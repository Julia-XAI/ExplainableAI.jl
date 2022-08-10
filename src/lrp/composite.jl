# A Composite is a container of primitives, which are sequentially applied
struct Composite{T<:Union{Tuple,AbstractVector}}
    primitives::T
end
Composite(rule::AbstractLRPRule, prims...) = Composite((GlobalRule(rule), prims...))
Composite(prims...) = Composite(prims)

const COMPOSITE_DEFAULT_RULE = ZeroRule()
function (c::Composite)(model)
    rules = Vector{AbstractLRPRule}(repeat([COMPOSITE_DEFAULT_RULE], length(model.layers)))
    for p in c.primitives
        p(rules, model.layers) # in-place modifies rules
    end
    return rules
end

# All primitives need to implement the following interfaces:
# * can be called with `layers` and `rules` of equal length and in-place modifies `rules`
# * implements `_range_string` that prints the positional range it is modifying rules on
abstract type AbstractCompositePrimitive end

###################
# Rule primitives #
###################
abstract type AbstractRulePrimitive <: AbstractCompositePrimitive end

"""
    LayerRule(n, rule)

Composite primitive that applies LRP-rule `rule` to the `n`-th layer in the model.

See [`Composite`](@ref) for an example.
"""
struct LayerRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    n::Int
    rule::R
end
(r::LayerRule)(rules, _layers) = (rules[r.n] = r.rule)

"""
    GlobalRule(rule)

Composite primitive that applies LRP-rule `rule` to all layers in the model.

See [`Composite`](@ref) for an example.
"""
struct GlobalRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end
(r::GlobalRule)(rules, _layers) = fill!(rules, r.rule)

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
function (r::RangeRule)(rules, _layers)
    for i in r.range
        rules[i] = r.rule
    end
end

"""
    FirstLayerRule(rule)

Composite primitive that applies LRP-rule `rule` to the first layer in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end
(r::FirstLayerRule)(rules, _layers) = (rules[1] = r.rule)

"""
    LastLayerRule(rule)

Composite primitive that applies LRP-rule `rule` to the last layer in the model.

See [`Composite`](@ref) for an example.
"""
struct LastLayerRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end
(r::LastLayerRule)(rules, _layers) = (rules[end] = r.rule)

######################
# TypeRule primitives #
######################
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
function (r::GlobalTypeRule)(rules, layers)
    return _range_rule_map!(rules, layers, r.map, 1:length(layers))
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
(r::RangeTypeRule)(rules, layers) = _range_rule_map!(rules, layers, r.map, r.range)

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
(r::FirstNTypeRule)(rules, layers) = _range_rule_map!(rules, layers, r.map, 1:(r.n))

"""
    LastNTypeRule(n, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the last `n` layers in the model.

See [`Composite`](@ref) for an example.
"""
struct LastNTypeRule{T<:AbstractVector{<:TypeRulePair}} <: AbstractTypeRulePrimitive
    n::Int
    map::T
end
function (r::LastNTypeRule)(rules, layers)
    l = length(layers)
    return _range_rule_map!(rules, layers, r.map, (l - r.n):l)
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
(r::FirstLayerTypeRule)(rules, layers) = _layer_rule_map!(rules, first(layers), r.map, 1)

"""
    LastLayerTypeRule(map)

Composite primitive that maps the type of the last layer of the model to LRP rules
based on a list of type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct LastLayerTypeRule{T<:AbstractVector{<:TypeRulePair}} <: AbstractTypeRulePrimitive
    map::T
end
function (r::LastLayerTypeRule)(rules, layers)
    return _layer_rule_map!(rules, last(layers), r.map, length(layers))
end
# Convenience constructors
GlobalTypeRule(ps::Vararg{<:TypeRulePair}) = GlobalTypeRule([ps...])
RangeTypeRule(r, ps::Vararg{<:TypeRulePair}) = RangeTypeRule(r, [ps...])
FirstLayerTypeRule(ps::Vararg{<:TypeRulePair}) = FirstLayerTypeRule([ps...])
LastLayerTypeRule(ps::Vararg{<:TypeRulePair}) = LastLayerTypeRule([ps...])
FirstNTypeRule(n, ps::Vararg{<:TypeRulePair}) = FirstNTypeRule(n, [ps...])
LastNTypeRule(n, ps::Vararg{<:TypeRulePair}) = LastNTypeRule(n, [ps...])

function _layer_rule_map!(rules, layer, map, index)
    for (Type, rule) in map
        if isa(layer, Type)
            rules[index] = rule
        end
    end
    return nothing
end
function _range_rule_map!(rules, layers, map, range)
    for (i, l) in enumerate(layers)
        i in range || continue
        _layer_rule_map!(rules, l, map, i)
    end
    return nothing
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
* [`LastNTypeRule`](@ref) for a `TypeRule` on the last `n` layers

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
