# A Composite is just a container of primitives, which are sequentially applied
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

See also [`Composite`](@ref).
"""
struct LayerRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    n::Int
    rule::R
end
(r::LayerRule)(rules, _layers) = (rules[r.n] = r.rule)

"""
    GlobalRule(rule)

Composite primitive that applies LRP-rule `rule` to all layers in the model.

See also [`Composite`](@ref).
"""
struct GlobalRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end
(r::GlobalRule)(rules, _layers) = fill!(rules, r.rule)

"""
    RangeRule(range, rule)

Composite primitive that applies LRP-rule `rule` to the specified positional `range`
of layers in the model.

See also [`Composite`](@ref).
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
    FirstRule(rule)

Composite primitive that applies LRP-rule `rule` to the first layer in the model.

See also [`Composite`](@ref).
"""
struct FirstRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end
(r::FirstRule)(rules, _layers) = (rules[1] = r.rule)

"""
    LastRule(rule)

Composite primitive that applies LRP-rule `rule` to the last layer in the model.

See also [`Composite`](@ref).
"""
struct LastRule{R<:AbstractLRPRule} <: AbstractRulePrimitive
    rule::R
end
(r::LastRule)(rules, _layers) = (rules[end] = r.rule)

######################
# RuleMap primitives #
######################
abstract type AbstractRuleMapPrimitive <: AbstractCompositePrimitive end
const TypeRulePair = Pair{<:Type,<:AbstractLRPRule}

"""
    GlobalRuleMap(map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map`.

See also [`Composite`](@ref).
"""
struct GlobalRuleMap{T<:AbstractVector{<:TypeRulePair}} <: AbstractRuleMapPrimitive
    map::T
end
(r::GlobalRuleMap)(rules, layers) = _map_rules!(rules, layers, r.map, 1:length(layers))

"""
    RangeRuleMap(range, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the specified `range` of layers in the model.

See also [`Composite`](@ref).
"""
struct RangeRuleMap{R<:AbstractRange,T<:AbstractVector{<:TypeRulePair}} <:
       AbstractRuleMapPrimitive
    range::R
    map::T
end
(r::RangeRuleMap)(rules, layers) = _map_rules!(rules, layers, r.map, r.range)

"""
    FirstNRuleMap(n, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the first `n` layers in the model.

See also [`Composite`](@ref).
"""
struct FirstNRuleMap{T<:AbstractVector{<:TypeRulePair}} <: AbstractRuleMapPrimitive
    n::Int
    map::T
end
(r::FirstNRuleMap)(rules, layers) = _map_rules!(rules, layers, r.map, 1:(r.n))

"""
    LastNRuleMap(n, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the last `n` layers in the model.

See also [`Composite`](@ref).
"""
struct LastNRuleMap{T<:AbstractVector{<:TypeRulePair}} <: AbstractRuleMapPrimitive
    n::Int
    map::T
end
function (r::LastNRuleMap)(rules, layers)
    l = length(layers)
    return _map_rules!(rules, layers, r.map, (l - r.n):l)
end

# Convenience constructors
GlobalRuleMap(ps::Vararg{<:TypeRulePair}) = GlobalRuleMap([ps...])
RangeRuleMap(r, ps::Vararg{<:TypeRulePair}) = RangeRuleMap(r, [ps...])
FirstNRuleMap(n, ps::Vararg{<:TypeRulePair}) = FirstNRuleMap(n, [ps...])
LastNRuleMap(n, ps::Vararg{<:TypeRulePair}) = LastNRuleMap(n, [ps...])

function _map_rules!(rules, layers, map, range)
    for (i, layer) in enumerate(layers)
        i in range || continue
        for (T, rule) in map
            if isa(layer, T)
                rules[i] = rule
            end
        end
    end
    return rules
end

"""
    Composite([default_rule=LRPZero], primitives...)

Automatically contructs a list of LRP-rules by sequentially applying composite primitives.

# Primitives
To apply a single rule, use:
* [`LayerRule`](@ref) to apply a rule to the `n`-th layer of a model
* [`GlobalRule`](@ref) to apply a rule to all layers of a model
* [`RangeRule`](@ref) to apply a rule to a positional range of layers of a model
* [`FirstRule`](@ref) to apply a rule to the first layer of a model
* [`LastRule`](@ref) to apply a rule to the last layer of a model

To apply a set of rules to multiple layers, use:
* [`GlobalRuleMap`](@ref) to apply a dictionary that maps layer types to LRP-rules
* [`RangeRuleMap`](@ref) for a `RuleMap` on generalized ranges
* [`FirstNRuleMap`](@ref) for a `RuleMap` on the first `n` layers of a model
* [`LastNRuleMap`](@ref) for a `RuleMap` on the last `n` layers

# Example
Using a flattened VGG11 model:
```julia-repl
julia> composite = Composite(
           GlobalRuleMap(
               ConvLayer => AlphaBetaRule(),
               Dense => EpsilonRule(),
               PoolingLayer => EpsilonRule(),
               DropoutLayer => PassRule(),
               ReshapingLayer => PassRule(),
           ),
           FirstNRuleMap(7, Conv => FlatRule()),
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
