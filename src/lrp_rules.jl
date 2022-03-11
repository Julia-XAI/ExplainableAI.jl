# Generic implementation of LRP according to [1, 2].
# LRP-rules are implemented as structs of type `AbstractLRPRule`.
# Through the magic of multiple dispatch, rule modifications such as LRP-γ and -ϵ
# can be implemented by dispatching on the functions `modify_params` & `modify_denominator`,
# which make use of the generalized LRP implementation shown in [1].
#
# If the relevance propagation falls outside of this scheme, custom functions
# ```julia
# (::MyLRPRule)(layer, aₖ, Rₖ₊₁) = ...
# (::MyLRPRule)(layer::MyLayer, aₖ, Rₖ₊₁) = ...
# (::AbstractLRPRule)(layer::MyLayer, aₖ, Rₖ₊₁) = ...
# ```
# that return `Rₖ` can be implemented.
# This is used for the ZBoxRule and for faster computations on common layers.
#
# References:
# [1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
# [2] W. Samek et al., Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications

abstract type AbstractLRPRule end

# This is the generic relevance propagation rule which is used for the 0, γ and ϵ rules.
# It can be extended for new rules via `modify_denominator` and `modify_params`.
# Since it uses autodiff, it is used as a fallback for layer types without custom implementation.
(rule::AbstractLRPRule)(layer, aₖ, Rₖ₊₁) = lrp_autodiff(rule, layer, aₖ, Rₖ₊₁)

function lrp_autodiff(rule, layer, aₖ, Rₖ₊₁)
    layerᵨ = _modify_layer(rule, layer)
    function fwpass(a)
        z = layerᵨ(a)
        s = Zygote.dropgrad(Rₖ₊₁ ./ modify_denominator(rule, z))
        return z ⋅ s
    end
    return aₖ .* gradient(fwpass, aₖ)[1] # Rₖ
end

# For linear layer types such as Dense layers, using autodiff is overkill.
(rule::AbstractLRPRule)(layer::Dense, aₖ, Rₖ₊₁) = lrp_dense(rule, layer, aₖ, Rₖ₊₁)

function lrp_dense(rule, l, aₖ, Rₖ₊₁)
    ρW, ρb = modify_params(rule, get_params(l)...)
    ãₖ₊₁ = modify_denominator(rule, ρW * aₖ + ρb)
    return @tullio Rₖ[j] := aₖ[j] * ρW[k, j] / ãₖ₊₁[k] * Rₖ₊₁[k]
end

# Other special cases that are dispatched on layer type:
(::AbstractLRPRule)(::DropoutLayer, aₖ, Rₖ₊₁) = Rₖ₊₁
(::AbstractLRPRule)(::ReshapingLayer, aₖ, Rₖ₊₁) = reshape(Rₖ₊₁, size(aₖ))

# To implement new rules, we can define two custom functions `modify_params` and `modify_denominator`.
# If this isn't done, the following fallbacks are used by default:
"""
    modify_params(rule, W, b)

Function that modifies weights and biases before applying relevance propagation.
"""
modify_params(::AbstractLRPRule, W, b) = (W, b) # general fallback

"""
    modify_denominator(rule, d)

Function that modifies zₖ on the forward pass, e.g. for numerical stability.
"""
modify_denominator(::AbstractLRPRule, d) = stabilize_denom(d; eps=1.0f-9) # general fallback

# This helper function applies `modify_params`:
_modify_layer(::AbstractLRPRule, layer) = layer # skip layers without modify_params
function _modify_layer(rule::AbstractLRPRule, layer::Union{Dense,Conv})
    return set_params(layer, modify_params(rule, get_params(layer)...)...)
end

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
function modify_params(r::GammaRule, W, b)
    ρW = W + r.γ * relu.(W)
    ρb = b + r.γ * relu.(b)
    return ρW, ρb
end

"""
    EpsilonRule(; ϵ=1f-6)

Constructor for LRP-``ϵ`` rule. Commonly used on middle layers.

Arguments:
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.
"""
struct EpsilonRule{T} <: AbstractLRPRule
    ϵ::T
    EpsilonRule(; ϵ=1.0f-6) = new{Float32}(ϵ)
end
modify_denominator(r::EpsilonRule, d) = stabilize_denom(d; eps=r.ϵ)

"""
    ZBoxRule()

Constructor for LRP-``z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
struct ZBoxRule <: AbstractLRPRule end

# The ZBoxRule requires its own implementation of relevance propagation.
(rule::ZBoxRule)(layer::Dense, aₖ, Rₖ₊₁) = lrp_zbox(layer, aₖ, Rₖ₊₁)
(rule::ZBoxRule)(layer::Conv, aₖ, Rₖ₊₁) = lrp_zbox(layer, aₖ, Rₖ₊₁)

function lrp_zbox(layer, aₖ, Rₖ₊₁)
    W, b = get_params(layer)
    l, h = fill.(extrema(aₖ), (size(aₖ),))

    layer⁺ = set_params(layer, max.(0, W), max.(0, b)) # W⁺, b⁺
    layer⁻ = set_params(layer, min.(0, W), min.(0, b)) # W⁻, b⁻

    # Forward pass
    function fwpass(a, l, h)
        f = layer(a)
        f⁺ = layer⁺(l)
        f⁻ = layer⁻(h)

        z = f - f⁺ - f⁻
        s = Zygote.dropgrad(safedivide(Rₖ₊₁, z; eps=1e-9))
        return z ⋅ s
    end
    c, cₗ, cₕ = gradient(fwpass, aₖ, l, h) # w.r.t. three inputs
    return aₖ .* c + l .* cₗ + h .* cₕ # Rₖ from backward pass
end
