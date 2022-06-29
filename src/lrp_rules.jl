# https://adrhill.github.io/ExplainableAI.jl/stable/generated/advanced_lrp/#How-it-works-internally
abstract type AbstractLRPRule end

# Generic LRP rule. Since it uses autodiff, it is used as a fallback for layer types
# without custom implementations.
function lrp!(Rₖ, rule::R, layer::L, aₖ, Rₖ₊₁) where {R<:AbstractLRPRule,L}
    check_compat(rule, layer)
    reset! = get_layer_resetter(rule, layer)
    modify_layer!(rule, layer)
    ãₖ₊₁, pullback = Zygote.pullback(layer, modify_input(rule, aₖ))
    Rₖ .= aₖ .* only(pullback(Rₖ₊₁ ./ modify_denominator(rule, ãₖ₊₁)))
    reset!()
    return nothing
end

# To implement new rules, define the following custom functions:
#   * `modify_input(rule, input)`
#   * `modify_denominator(rule, d)`
#   * `check_compat(rule, layer)`
#   * `modify_param!(rule, param)` or `modify_layer!(rule, layer)`,
#     the latter overriding the former
#
# The following fallbacks are used by default:
"""
    modify_input(rule, input)

Modify input activation before computing relevance propagation.
"""
@inline modify_input(rule, input) = input # general fallback

"""
    modify_denominator(rule, d)

Modify denominator ``z`` for numerical stability on the forward pass.
"""
@inline modify_denominator(rule, d) = stabilize_denom(d, 1.0f-9) # general fallback

"""
    check_compat(rule, layer)

Check compatibility of LRP-Rule with layer.
Returns nothing if checks passed, otherwise throws `ArgumentError`.
"""
@inline check_compat(rule, layer) = nothing # general fallback

"""
    modify_layer!(rule, layer)

In-place modify layer parameters by calling `modify_param!` before computing relevance
propagation.

## Note
When implementing a custom `modify_layer!` function, `modify_param!` will not be called.
"""
modify_layer!(rule, layer) = nothing
for L in WeightBiasLayers
    @eval function modify_layer!(rule::R, layer::$L) where {R}
        if has_weight_and_bias(layer)
            modify_param!(rule, layer.weight)
            modify_bias!(rule, layer.bias)
        end
        return nothing
    end
end

"""
    modify_param!(rule, W)
    modify_param!(rule, b)

Inplace-modify parameters before computing the relevance.
"""
@inline modify_param!(rule, param) = nothing # general fallback

# Useful presets:
modify_param!(::Val{:mask_positive}, p) = (p .= max.(zero(eltype(p)), p), return nothing)
modify_param!(::Val{:mask_negative}, p) = (p .= min.(zero(eltype(p)), p), return nothing)

# Internal wrapper functions for bias-free layers.
@inline modify_bias!(rule::R, b) where {R} = modify_param!(rule, b)
@inline modify_bias!(rule, b::Flux.Zeros) = nothing # skip if bias=Flux.Zeros (Flux <= v0.12)
@inline function modify_bias!(rule, b::Bool) # skip if bias=false (Flux >= v0.13)
    @assert b == false
    return nothing
end

# Internal function that resets parameters by capturing them in a closure.
# Returns a function `reset!` that resets the parameters to their original state when called.
function get_layer_resetter(rule, layer)
    !has_weight_and_bias(layer) && return Returns(nothing)
    W = deepcopy(layer.weight)
    b = deepcopy(layer.bias)

    function reset!()
        layer.weight .= W
        isa(layer.bias, AbstractArray) && (layer.bias .= b)
        return nothing
    end
    return reset!
end

"""
    ZeroRule()

Constructor for LRP-0 rule. Commonly used on upper layers.
"""
struct ZeroRule <: AbstractLRPRule end

"""
    GammaRule([γ=0.25])

Constructor for LRP-``γ`` rule. Commonly used on lower layers.

Arguments:
- `γ`: Optional multiplier for added positive weights, defaults to 0.25.
"""
struct GammaRule{T} <: AbstractLRPRule
    γ::T
    GammaRule(γ=0.25f0) = new{Float32}(γ)
end
function modify_param!(r::GammaRule, param::AbstractArray{T}) where {T}
    γ = convert(T, r.γ)
    param .+= γ * relu.(param)
    return nothing
end
check_compat(rule::GammaRule, layer) = require_weight_and_bias(rule, layer)

"""
    EpsilonRule([ϵ=1.0f-6])

Constructor for LRP-``ϵ`` rule. Commonly used on middle layers.

Arguments:
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.
"""
struct EpsilonRule{T} <: AbstractLRPRule
    ϵ::T
    EpsilonRule(ϵ=1.0f-6) = new{Float32}(ϵ)
end
modify_denominator(r::EpsilonRule, d) = stabilize_denom(d, r.ϵ)

"""
    ZBoxRule(low, high)

Constructor for LRP-``z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.

The parameters `low` and `high` should be set to the lower and upper bounds of the input features,
e.g. `0.0` and `1.0` for raw image data.
It is also possible to provide two arrays of that match the input size.
"""
struct ZBoxRule{T} <: AbstractLRPRule
    low::T
    high::T
end

# The ZBoxRule requires its own implementation of relevance propagation.
function lrp!(Rₖ, rule::ZBoxRule, layer::L, aₖ, Rₖ₊₁) where {L}
    require_weight_and_bias(rule, layer)
    reset! = get_layer_resetter(rule, layer)

    l = zbox_input(aₖ, rule.low)
    h = zbox_input(aₖ, rule.high)

    # Compute pullback for W, b
    aₖ₊₁, pullback = Zygote.pullback(layer, aₖ)

    # Compute pullback for W⁺, b⁺
    modify_layer!(Val{:mask_positive}, layer)
    aₖ₊₁⁺, pullback⁺ = Zygote.pullback(layer, l)
    reset!()

    # Compute pullback for W⁻, b⁻
    modify_layer!(Val{:mask_negative}, layer)
    aₖ₊₁⁻, pullback⁻ = Zygote.pullback(layer, h)
    reset!()

    y = Rₖ₊₁ ./ modify_denominator(rule, aₖ₊₁ - aₖ₊₁⁺ - aₖ₊₁⁻)
    Rₖ .= aₖ .* only(pullback(y)) - l .* only(pullback⁺(y)) - h .* only(pullback⁻(y))
    return nothing
end

zbox_input(in::AbstractArray{T}, c::Real) where {T} = fill(convert(T, c), size(in))
function zbox_input(in::AbstractArray{T}, A::AbstractArray) where {T}
    @assert size(A) == size(in)
    return convert.(T, A)
end

# Other special cases that are dispatched on layer type:
const LRPRules = (ZeroRule, EpsilonRule, GammaRule, ZBoxRule)
for R in LRPRules
    @eval lrp!(Rₖ, ::$R, ::DropoutLayer, aₖ, Rₖ₊₁) = (Rₖ .= Rₖ₊₁)
    @eval lrp!(Rₖ, ::$R, ::ReshapingLayer, aₖ, Rₖ₊₁) = (Rₖ .= reshape(Rₖ₊₁, size(aₖ)))
end
# Fast implementation for Dense layer using Tullio.jl's einsum notation:
for R in (ZeroRule, EpsilonRule, GammaRule)
    @eval function lrp!(Rₖ, rule::$R, layer::Dense, aₖ, Rₖ₊₁)
        reset! = get_layer_resetter(rule, layer)
        modify_layer!(rule, layer)
        ãₖ₊₁ = modify_denominator(rule, layer(modify_input(rule, aₖ)))
        @tullio Rₖ[j, b] = aₖ[j, b] * layer.weight[k, j] * Rₖ₊₁[k, b] / ãₖ₊₁[k, b]
        reset!()
        return nothing
    end
end

# Rules that don't modify params can optionally be added here for extra performance
get_layer_resetter(::ZeroRule, l) = Returns(nothing)
get_layer_resetter(::EpsilonRule, l) = Returns(nothing)
