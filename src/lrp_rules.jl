
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
const LRPRuleset = AbstractVector{<:AbstractLRPRule}
"""
    modify_params!(w, rule)

Function that modifies weights and biases before applying relevance propagation.
"""
modify_params(::AbstractLRPRule, p) = p

"""
    modify_denominator!(d, rule)

Function that modifies zₖ on the forward pass, e.g. for numerical stability.
"""
modify_denominator(::AbstractLRPRule, d) = stabilize_denom(d; eps=1f-9)

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
modify_params(r::GammaRule, p) = p + r.γ * relu.(p)

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


LinearLayer = Union{Dense,Conv,MaxPool}

# This is the generic relevance propagation rule which is used for the 0, γ and ϵ rules.
# It can be extended for new rules via `modify_params` and `modify_denominator`.
function (rule::AbstractLRPRule)(
    layer::L, aₖ::AbstractArray, Rₖ₊₁::AbstractArray
) where {L<:LinearLayer}
    # Forward pass
    W, b = get_weights(layer)
    ρW = modify_params(rule, W)
    ρb = modify_params(rule, b)

    function fwpass(a)
        z = ρW * a + ρb
        s = Zygote.dropgrad(Rₖ₊₁ ./ modify_denominator(rule, z))
        return z ⋅ s
    end
    c = gradient(fwpass, aₖ)[1]

    # Backward pass
    Rₖ = aₖ .* c
    return Rₖ
end

"""
    ZBoxRule()

Constructor for LRP-`z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
struct ZBoxRule <: AbstractLRPRule end

# The ZBoxRule requires its own implementation of relevance propagation.
function (::ZBoxRule)(
    layer::L, aₖ::AbstractArray, Rₖ₊₁::AbstractArray
) where {L<:LinearLayer}
    W, b = get_weights(layer)
    W⁻ = min.(0, W)
    W⁺ = max.(0, W)
    b⁻ = min.(0, b)
    b⁺ = max.(0, b)

    onemat = ones(eltype(aₖ), size(aₖ))
    l = onemat * minimum(aₖ)
    h = onemat * maximum(aₖ)

    # Forward pass
    function fwpass(a, l, h)
        f = W * a + b
        f⁺ = W⁺ * l + b⁺
        f⁻ = W⁻ * h + b⁻

        z = f - f⁺ - f⁻
        s = Zygote.dropgrad(Rₖ₊₁ ./ stabilize_denom(z; eps=1e-9))
        return z ⋅ s
    end
    c, cₗ, cₕ = gradient(fwpass, aₖ, l, h) # w.r.t. three inputs

    # Backward pass
    Rₖ = aₖ .* c + l .* cₗ + h .* cₕ
    return Rₖ
end
