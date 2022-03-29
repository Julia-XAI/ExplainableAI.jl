# https://adrhill.github.io/ExplainableAI.jl/stable/generated/advanced_lrp/#How-it-works-internally
abstract type AbstractLRPRule end

# Generic LRP rule. Since it uses autodiff, it is used as a fallback for layer types without custom implementation.
function lrp!(rule::R, layer::L, Rₖ, aₖ, Rₖ₊₁) where {R<:AbstractLRPRule,L}
    lrp_autodiff!(rule, layer, Rₖ, aₖ, Rₖ₊₁)
    return nothing
end

function lrp_autodiff!(
    rule::R, layer::L, Rₖ::T1, aₖ::T1, Rₖ₊₁::T2
) where {R<:AbstractLRPRule,L,T1,T2}
    layerᵨ = modify_layer(rule, layer)
    c::T1 = only(
        gradient(aₖ) do a
            z::T2 = layerᵨ(a)
            s = Zygote.@ignore Rₖ₊₁ ./ modify_denominator(rule, z)
            z ⋅ s
        end,
    )
    Rₖ .= aₖ .* c
    return nothing
end

# For linear layer types such as Dense layers, using autodiff is overkill.
function lrp!(rule::R, layer::Dense, Rₖ, aₖ, Rₖ₊₁) where {R<:AbstractLRPRule}
    lrp_dense!(rule, layer, Rₖ, aₖ, Rₖ₊₁)
    return nothing
end

function lrp_dense!(rule::R, l, Rₖ, aₖ, Rₖ₊₁) where {R<:AbstractLRPRule}
    ρW, ρb = modify_params(rule, get_params(l)...)
    ãₖ₊₁ = modify_denominator(rule, ρW * aₖ .+ ρb)
    @tullio Rₖ[j, b] = aₖ[j, b] * ρW[k, j] / ãₖ₊₁[k, b] * Rₖ₊₁[k, b]
    return nothing
end

# Other special cases that are dispatched on layer type:
lrp!(::AbstractLRPRule, ::DropoutLayer, Rₖ, aₖ, Rₖ₊₁) = (Rₖ .= Rₖ₊₁)
lrp!(::AbstractLRPRule, ::ReshapingLayer, Rₖ, aₖ, Rₖ₊₁) = (Rₖ .= reshape(Rₖ₊₁, size(aₖ)))

# To implement new rules, we can define two custom functions `modify_params` and `modify_denominator`.
# If this isn't done, the following fallbacks are used by default:
"""
    modify_params(rule, W, b)

Function that modifies weights and biases before applying relevance propagation.
Returns modified weights and biases as a tuple `(ρW, ρb)`.
"""
modify_params(::AbstractLRPRule, W, b) = (W, b) # general fallback

"""
    modify_denominator(rule, d)

Function that modifies zₖ on the forward pass, e.g. for numerical stability.
"""
modify_denominator(::AbstractLRPRule, d) = stabilize_denom(d, 1.0f-9) # general fallback

"""
    modify_layer(rule, layer)

Function that modifies a layer before applying relevance propagation.
Returns a new, modified layer.
"""
modify_layer(::AbstractLRPRule, layer) = layer # skip layers without modify_params
function modify_layer(rule::R, layer::L) where {R<:AbstractLRPRule,L<:Union{Dense,Conv}}
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
modify_denominator(r::EpsilonRule, d) = stabilize_denom(d, r.ϵ)

"""
    ZBoxRule()

Constructor for LRP-``z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
struct ZBoxRule <: AbstractLRPRule end

# The ZBoxRule requires its own implementation of relevance propagation.
lrp!(::ZBoxRule, layer::Dense, Rₖ, aₖ, Rₖ₊₁) = lrp_zbox!(layer, Rₖ, aₖ, Rₖ₊₁)
lrp!(::ZBoxRule, layer::Conv, Rₖ, aₖ, Rₖ₊₁) = lrp_zbox!(layer, Rₖ, aₖ, Rₖ₊₁)

function lrp_zbox!(layer::L, Rₖ::T1, aₖ::T1, Rₖ₊₁::T2) where {L,T1,T2}
    W, b = get_params(layer)
    l, h = fill.(extrema(aₖ), (size(aₖ),))

    layer⁺ = set_params(layer, max.(0, W), max.(0, b)) # W⁺, b⁺
    layer⁻ = set_params(layer, min.(0, W), min.(0, b)) # W⁻, b⁻

    c::T1, cₗ::T1, cₕ::T1 = gradient(aₖ, l, h) do a, l, h
        f::T2 = layer(a)
        f⁺::T2 = layer⁺(l)
        f⁻::T2 = layer⁻(h)

        z = f - f⁺ - f⁻
        s = Zygote.@ignore safedivide(Rₖ₊₁, z, 1e-9)
        z ⋅ s
    end
    Rₖ .= aₖ .* c + l .* cₗ + h .* cₕ
    return nothing
end
