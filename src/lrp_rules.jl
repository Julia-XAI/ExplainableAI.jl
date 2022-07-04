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
modify_input(rule, input) = input # general fallback

"""
    modify_denominator(rule, d)

Modify denominator ``z`` for numerical stability on the forward pass.
"""
modify_denominator(rule, d) = stabilize_denom(d, 1.0f-9) # general fallback

"""
    check_compat(rule, layer)

Check compatibility of a LRP-Rule with layer type.

## Note
When implementing a custom `check_compat` function, return `nothing` if checks passed,
otherwise throw an `ArgumentError`.
"""
check_compat(rule, layer) = require_weight_and_bias(rule, layer)

"""
    modify_layer!(rule, layer; ignore_bias=false)

In-place modify layer parameters by calling `modify_param!` before computing relevance
propagation.

## Note
When implementing a custom `modify_layer!` function, `modify_param!` will not be called.
"""
function modify_layer!(rule::R, layer::L; ignore_bias=false) where {R,L}
    !has_weight_and_bias(layer) && return nothing # skip all
    modify_weight!(rule, layer.weight)

    # Checks that skip bias modification:
    ignore_bias && return nothing
    isa(layer.bias, Flux.Zeros) && return nothing # skip if bias=Flux.Zeros (Flux <= v0.12)
    isa(layer.bias, Bool) && !layer.bias && return nothing  # skip if bias=false (Flux >= v0.13)

    modify_bias!(rule, layer.bias)
    return nothing
end
modify_weight!(rule::R, W) where {R} = modify_param!(rule, W)
modify_bias!(rule::R, b) where {R} = modify_param!(rule, b)

"""
    modify_param!(rule, W)
    modify_param!(rule, b)

Inplace-modify parameters before computing the relevance.
"""
modify_param!(rule, param) = nothing # general fallback

# Useful presets that allow us to work around bias-free layers:
modify_param!(::Val{:keep_positive}, p) = keep_positive!(p)
modify_param!(::Val{:keep_negative}, p) = keep_negative!(p)

modify_weight!(::Val{:keep_positive_zero_bias}, W) = keep_positive!(W)
modify_bias!(::Val{:keep_positive_zero_bias}, b) = fill!(b, zero(eltype(b)))

modify_weight!(::Val{:keep_negative_zero_bias}, W) = keep_negative!(W)
modify_bias!(::Val{:keep_negative_zero_bias}, b) = fill!(b, zero(eltype(b)))

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

LRP-0 rule. Commonly used on upper layers.

# References
[1]: S. Bach et al., On Pixel-Wise Explanations for Non-Linear Classifier Decisions by
    Layer-Wise Relevance Propagation
"""
struct ZeroRule <: AbstractLRPRule end
check_compat(::ZeroRule, layer) = nothing

# Optimization to save allocations since weights don't need to be reset:
get_layer_resetter(::ZeroRule, layer) = Returns(nothing)

"""
    EpsilonRule([ϵ=1.0f-6])

LRP-``ϵ`` rule. Commonly used on middle layers.

Arguments:
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.

# References
[1]: S. Bach et al., On Pixel-Wise Explanations for Non-Linear Classifier Decisions by
    Layer-Wise Relevance Propagation
"""
struct EpsilonRule{T} <: AbstractLRPRule
    ϵ::T
    EpsilonRule(ϵ=1.0f-6) = new{Float32}(ϵ)
end
modify_denominator(r::EpsilonRule, d) = stabilize_denom(d, r.ϵ)
check_compat(::EpsilonRule, layer) = nothing

# Optimization to save allocations since weights don't need to be reset:
get_layer_resetter(::EpsilonRule, layer) = Returns(nothing)

"""
    GammaRule([γ=0.25])

LRP-``γ`` rule. Commonly used on lower layers.

Arguments:
- `γ`: Optional multiplier for added positive weights, defaults to `0.25`.

# References
[1]: G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
"""
struct GammaRule{T} <: AbstractLRPRule
    γ::T
    GammaRule(γ=0.25f0) = new{Float32}(γ)
end
function modify_param!(r::GammaRule, param::AbstractArray{T}) where {T}
    γ = convert(T, r.γ)
    param .+= γ .* relu.(param)
    return nothing
end

"""
    WSquareRule()

LRP-``W^2`` rule. Commonly used on the first layer when values are unbounded.

# References
[1]: G. Montavon et al., Explaining nonlinear classification decisions with deep Taylor decomposition
"""
struct WSquareRule <: AbstractLRPRule end
modify_param!(::WSquareRule, p) = p .^= 2
modify_input(::WSquareRule, input) = ones_like(input)

"""
    FlatRule()

LRP-Flat rule. Similar to the [`WSquareRule`](@ref), but with all parameters set to one.

# References
[1]: S. Lapuschkin et al., Unmasking Clever Hans predictors and assessing what machines really learn
"""
struct FlatRule <: AbstractLRPRule end
modify_param!(::FlatRule, p) = fill!(p, 1)
modify_input(::FlatRule, input) = ones_like(input)

"""
    PassRule()

Pass-through rule. Passes relevance through to the lower layer.
Supports reshaping layers.
"""
struct PassRule <: AbstractLRPRule end
function lrp!(Rₖ, ::PassRule, layer, aₖ, Rₖ₊₁)
    if size(aₖ) == size(Rₖ₊₁)
        Rₖ .= Rₖ₊₁
    end
    Rₖ .= reshape(Rₖ₊₁, size(aₖ))
    return nothing
end
# No extra checks as reshaping operation will throw an error if layer isn't compatible:
check_compat(::PassRule, layer) = nothing

"""
    ZBoxRule(low, high)

LRP-``z^{\\mathcal{B}}``-rule. Commonly used on the first layer for pixel input.

The parameters `low` and `high` should be set to the lower and upper bounds of the input features,
e.g. `0.0` and `1.0` for raw image data.
It is also possible to provide two arrays of that match the input size.

## References
[1]: G. Montavon et al., Explaining nonlinear classification decisions with deep Taylor decomposition
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
    modify_layer!(Val(:keep_positive), layer)
    aₖ₊₁⁺, pullback⁺ = Zygote.pullback(layer, l)
    reset!()

    # Compute pullback for W⁻, b⁻
    modify_layer!(Val(:keep_negative), layer)
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

"""
    AlphaBetaRule(alpha, beta)
    AlphaBetaRule([alpha=2.0], [beta=1.0])

LRP-``\alpha\beta`` rule. Weights positive and negative contributions according to the
parameters `alpha` and `beta` respectively. The difference `alpha - beta` must be equal one.
Commonly used on lower layers.

Arguments:
- `alpha`: Multiplier for the positive output term, defaults to `2.0`.
- `beta`: Multiplier for the negative output term, defaults to `1.0`.

# References
[1]: S. Bach et al., On Pixel-Wise Explanations for Non-Linear Classifier Decisions by
    Layer-Wise Relevance Propagation
[2]: G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
"""
struct AlphaBetaRule{T} <: AbstractLRPRule
    α::T
    β::T
    function AlphaBetaRule(alpha=2.0f0, beta=1.0f0)
        alpha < 0 && throw(ArgumentError("Parameter `alpha` must be ≥0."))
        beta < 0 && throw(ArgumentError("Parameter `beta` must be ≥0."))
        !isone(alpha - beta) && throw(ArgumentError("`alpha - beta` must be equal one."))
        return new{Float32}(alpha, beta)
    end
end

# The AlphaBetaRule requires its own implementation of relevance propagation.
function lrp!(Rₖ, rule::AlphaBetaRule, layer::L, aₖ, Rₖ₊₁) where {L}
    require_weight_and_bias(rule, layer)
    reset! = get_layer_resetter(rule, layer)

    aₖ⁺ = keep_positive(aₖ)
    aₖ⁻ = keep_negative(aₖ)

    modify_layer!(Val(:keep_positive), layer)
    out_1, pullback_1 = Zygote.pullback(layer, aₖ⁺)
    reset!()
    modify_layer!(Val(:keep_negative_zero_bias), layer)
    out_2, pullback_2 = Zygote.pullback(layer, aₖ⁻)
    reset!()
    modify_layer!(Val(:keep_negative), layer)
    out_3, pullback_3 = Zygote.pullback(layer, aₖ⁺)
    reset!()
    modify_layer!(Val(:keep_positive_zero_bias), layer)
    out_4, pullback_4 = Zygote.pullback(layer, aₖ⁻)
    reset!()

    y_α = Rₖ₊₁ ./ modify_denominator(rule, out_1 + out_2)
    y_β = Rₖ₊₁ ./ modify_denominator(rule, out_3 + out_4)
    Rₖ .=
        rule.α .* (aₖ⁺ .* only(pullback_1(y_α)) + aₖ⁻ .* only(pullback_2(y_α))) .-
        rule.β .* (aₖ⁺ .* only(pullback_3(y_β)) + aₖ⁻ .* only(pullback_4(y_β)))
    return nothing
end

# Special cases for rules that don't modify params for extra performance:
for R in (ZeroRule, EpsilonRule)
    for L in (DropoutLayer, ReshapingLayer)
        @eval lrp!(Rₖ, ::$R, l::$L, aₖ, Rₖ₊₁) = lrp!(Rₖ, PassRule(), l, aₖ, Rₖ₊₁)
    end
end

# Fast implementation for Dense layer using Tullio.jl's einsum notation:
for R in (ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule)
    @eval function lrp!(Rₖ, rule::$R, layer::Dense, aₖ, Rₖ₊₁)
        reset! = get_layer_resetter(rule, layer)
        modify_layer!(rule, layer)
        ãₖ₊₁ = modify_denominator(rule, layer(modify_input(rule, aₖ)))
        @tullio Rₖ[j, b] = aₖ[j, b] * layer.weight[k, j] * Rₖ₊₁[k, b] / ãₖ₊₁[k, b]
        reset!()
        return nothing
    end
end
