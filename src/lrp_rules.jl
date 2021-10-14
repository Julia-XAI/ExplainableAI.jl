# Generic implementation of LRP according to [1, 2].
# LRP-rules are implemented as structs of type `AbstractLRPRule`.
# Through the magic of multiple dispatch, rule modifications such as LRP-γ and -ϵ
# can be implemented by dispatching on the functions `modify_params` & `modify_denominator`,
# which make use of the generalized LRP implementation shown in [1].
#
# If the relevance propagation falls outside of this scheme, a custom function
# ```julia
# (::MyLRPRule)(layer, aₖ, Rₖ₊₁) = ...
# ```
# can be implemented. This is used for the ZBoxRule.
#
# References:
# [1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
# [2] W. Samek et al., Explaining Deep Neural Networks and Beyond:
#     A Review of Methods and Applications

abstract type AbstractLRPRule end

# This is the generic relevance propagation rule which is used for the 0, γ and ϵ rules.
# It can be extended for new rules via `modify_denominator` and `modify_layer`,
# which in turn uses `modify_params`.
function (rule::AbstractLRPRule)(layer::Union{Dense,Conv,MaxPool,MeanPool}, aₖ, Rₖ₊₁)
    layerᵨ = modify_layer(rule, layer)
    function fwpass(a)
        z = layerᵨ(a)
        s = Zygote.dropgrad(Rₖ₊₁ ./ modify_denominator(rule, z))
        return z ⋅ s
    end
    c = gradient(fwpass, aₖ)[1]
    Rₖ = aₖ .* c
    return Rₖ
end

# Special cases are dispatched on layer type:
(rule::AbstractLRPRule)(::Dropout, aₖ, Rₖ₊₁) = Rₖ₊₁
(rule::AbstractLRPRule)(::typeof(Flux.flatten), aₖ, Rₖ₊₁) = reshape(Rₖ₊₁, size(aₖ))

"""
    modify_params!(rule, W, b)

Function that modifies weights and biases before applying relevance propagation.
"""
modify_params(::AbstractLRPRule, W, b) = (W, b) # general fallback

"""
    modify_layer(rule, layer)

Applies `modify_params` to layer if it has parameters
"""
modify_layer(::AbstractLRPRule, l) = l # skip layers without params
function modify_layer(rule::AbstractLRPRule, l::Union{Dense, Chain})
    W, b = get_weights(l)
    ρW, ρb = modify_params(rule, W, b)
    return set_weights(l, ρW, ρb)
end

"""
    modify_denominator!(d, rule)

Function that modifies zₖ on the forward pass, e.g. for numerical stability.
"""
modify_denominator(::AbstractLRPRule, d) = stabilize_denom(d; eps=1f-9) # general fallback

"""
    ZeroRule()

Constructor for LRP-0 rule. Commonly used on upper layers.
"""
struct ZeroRule <: AbstractLRPRule end

"""
    GammaRule(; γ=0.25)

Constructor for LRP-``γ`` rule. Commonly used on lower layers.

Arguments:
- `γ`: Optional multiplier for added positive weights, defaults to 0.25.
"""
struct GammaRule{T} <: AbstractLRPRule
    γ::T
    GammaRule(; γ=0.25) = new{Float32}(γ)
end
modify_params(r::GammaRule, W, b) = (W + r.γ * relu.(W), b)

"""
    EpsilonRule(; ε=1f-6)

Constructor for LRP-``ϵ`` rule. Commonly used on middle layers.

Arguments:
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.
"""
struct EpsilonRule{T} <: AbstractLRPRule
    ϵ::T
    EpsilonRule(; ε=1f-6) = new{Float32}(ε)
end
modify_denominator(r::EpsilonRule, d) = stabilize_denom(d; eps=1f-6)

"""
    ZBoxRule()

Constructor for LRP-`z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
struct ZBoxRule <: AbstractLRPRule end

# The ZBoxRule requires its own implementation of relevance propagation.
function (rule::ZBoxRule)(layer::L, aₖ, Rₖ₊₁) where {L<:Dense}
    layer, layer⁺, layer⁻ = modify_layer(rule, layer)

    onemat = ones(eltype(aₖ), size(aₖ))
    l = onemat * minimum(aₖ)
    h = onemat * maximum(aₖ)

    # Forward pass
    function fwpass(a, l, h)
        f = layer(a)
        f⁺ = layer⁺(l)
        f⁻ = layer⁻(h)

        z = f - f⁺ - f⁻
        s = Zygote.dropgrad(Rₖ₊₁ ./ stabilize_denom(z; eps=1e-9))
        return z ⋅ s
    end
    c, cₗ, cₕ = gradient(fwpass, aₖ, l, h) # w.r.t. three inputs

    # Backward pass
    Rₖ = aₖ .* c + l .* cₗ + h .* cₕ
    return Rₖ
end

function modify_layer(::ZBoxRule, l::Union{Dense, Chain})
    W, b = get_weights(l)
    W⁻ = min.(0, W)
    W⁺ = max.(0, W)
    b⁻ = min.(0, b)
    b⁺ = max.(0, b)

    l⁺ = set_weights(l, W⁺, b⁺)
    l⁻ = set_weights(l, W⁻, b⁻)
    return l, l⁺, l⁻
end
