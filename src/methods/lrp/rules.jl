
# Generic implementation of LRP according to [1, 2].
# LRP-rules are implemented as structs of type `AbstractLRPRule`.
# Through the magic of multiple dispatch, rule modifications such as LRP-γ and -ϵ
# can be implemented by dispatching on the functions `modify_params` & `modify_denominator`,
# which make use of the generalized LRP implementation shown in [1].

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

"""
    EpsilonRule(; ε=1f-6)

Constructor for LRP-``ϵ`` rule. Commonly used on middle layers.

Arguments:
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.
"""
struct EpsilonRule{T} <: AbstractLRPRule
    ε::T
    EpsilonRule(; ε=1f-6) = new{Float32}(ε)
end

"""
    ZBoxRule()

Constructor for LRP-`z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
struct ZBoxRule <: AbstractLRPRule end

"""
    modify_params!(w, rule)

Function that modifies weights and biases before applying relevance propagation.
"""
modify_params(w, ::AbstractReverseRule) = w
modify_params(w, r::GammaRule) = w + r.γ * relu.(w)

"""
    modify_denominator!(d, rule)

Function that increments zₖ on the forward pass for numerical stability.
"""
modify_denominator(d, ::AbstractReverseRule) = d
modify_denominator(d, r::EpsilonRule) = (1 + r.ϵ * std(d)) * d

LinearLayer = Union{Dense,Conv,MaxPool}

function (rule::AbstractLRPRule)(
    layer::L, aₖ::AbstractArray, Rₖ₊₁::AbstractArray
) where {L<:LinearLayer}
    # Forward pass
    W, b = get_weights(layer)
    ρW = modify_params(W, rule)

    function fwpass(a)
        z = modify_denominator(ρW * a + b, rule)
        s = Rₖ₊₁ ./ (z .+ 1e-9)
        return z ⋅ s
    end
    c = gradient(fwpass, aₖ)[1]

    # Backward pass
    Rₖ = aₖ .* c
    return Rₖ
end

# The ZBoxRule requires its own implementation of relevance propagation.
function (::ZBoxRule)(
    layer::L, aₖ::AbstractArray, Rₖ₊₁::AbstractArray
) where {L<:LinearLayer}
    W, b = get_weights(layer)
    W⁻ = min.(0, W)
    W⁺ = max.(0, W)
    b⁻ = min.(0, b)
    b⁺ = max.(0, b)

    onemat = ones(T, size(aₖ))
    l = onemat * minimum(aₖ)
    h = onemat * maximum(aₖ)

    # Forward pass
    function fwpass(a)
        f = W * a + b
        f⁺ = W⁺ * l + b⁺
        f⁻ = W⁻ * h + b⁻

        z = f - f⁺ - f⁻
        s = Rₖ₊₁ ./ (z .+ 1e-9)
        return z ⋅ s
    end

    dfw(a) = gradient(fwpass, a)[1]

    c = dfw(aₖ)
    cₗ = dfw(l)
    cₕ = dfw(h)

    println(c, cₗ, cₕ)

    # Backward pass
    Rₖ = aₖ .* c + l .* cₗ + h .* cₕ
    return Rₖ
end

# useful helper function to work around Flux.Zeros
function get_weights(layer)
    W = layer.weight
    if typeof(layer.bias) <: Flux.Zeros
        b = zeros(eltype(W), size(W, 1))
    else
        b = layer.bias
    end
    return W, b
end
