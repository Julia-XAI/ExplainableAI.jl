# https://adrhill.github.io/ExplainableAI.jl/stable/generated/advanced_lrp/#How-it-works-internally
abstract type AbstractLRPRule end

# Bibliography
const REF_BACH_LRP = "S. Bach et al., *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*"
const REF_LAPUSCHKIN_CLEVER_HANS = "S. Lapuschkin et al., *Unmasking Clever Hans predictors and assessing what machines really learn*"
const REF_MONTAVON_DTD = "G. Montavon et al., *Explaining Nonlinear Classification Decisions with Deep Taylor Decomposition*"
const REF_MONTAVON_OVERVIEW = "G. Montavon et al., *Layer-Wise Relevance Propagation: An Overview*"

# Generic LRP rule. Since it uses autodiff, it is used as a fallback for layer types
# without custom implementations.
function lrp!(Rₖ, rule::R, layer::L, aₖ, Rₖ₊₁) where {R<:AbstractLRPRule,L}
    check_compat(rule, layer)
    reset! = get_layer_resetter(rule, layer)
    modify_layer!(rule, layer)
    ãₖ = modify_input(rule, aₖ)
    zₖ₊₁, pullback = Zygote.pullback(preactivation(layer), ãₖ)
    Rₖ .= ãₖ .* only(pullback(Rₖ₊₁ ./ modify_denominator(rule, zₖ₊₁)))
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
    isa(layer.bias, Bool) && !layer.bias && return nothing

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

LRP-``0`` rule. Commonly used on upper layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i \\frac{w_{ij}a_j^k}{\\sum_l w_{il}a_l^k+b_i} R_i^{k+1}
```

# References
- $REF_BACH_LRP
"""
struct ZeroRule <: AbstractLRPRule end
check_compat(::ZeroRule, layer) = nothing

# Optimization to save allocations since weights don't need to be reset:
get_layer_resetter(::ZeroRule, layer) = Returns(nothing)

"""
    EpsilonRule([ϵ=1.0f-6])

LRP-``ϵ`` rule. Commonly used on middle layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{w_{ij}a_j^k}{\\epsilon +\\sum_{l}w_{il}a_l^k+b_i} R_i^{k+1}
```

# Optional arguments
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.

# References
- $REF_BACH_LRP
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

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{(w_{ij}+\\gamma w_{ij}^+)a_j^k}
    {\\sum_l(w_{il}+\\gamma w_{il}^+)a_l^k+b_i} R_i^{k+1}
```

# Optional arguments
- `γ`: Optional multiplier for added positive weights, defaults to `0.25`.

# References
- $REF_MONTAVON_OVERVIEW
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

LRP-``w²`` rule. Commonly used on the first layer when values are unbounded.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{w_{ij}^2}{\\sum_l w_{il}^2} R_i^{k+1}
```

# References
- $REF_MONTAVON_DTD
"""
struct WSquareRule <: AbstractLRPRule end
modify_weight!(::WSquareRule, w) = w .^= 2
modify_bias!(::WSquareRule, b) = fill!(b, 0)
modify_input(::WSquareRule, input) = ones_like(input)

"""
    FlatRule()

LRP-Flat rule. Similar to the [`WSquareRule`](@ref), but with all weights set to one
and all bias terms set to zero.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{1}{\\sum_l 1} R_i^{k+1} = \\sum_i\\frac{1}{n_i} R_i^{k+1}
```
where ``n_i`` is the number of input neurons connected to the output neuron at index ``i``.

# References
- $REF_LAPUSCHKIN_CLEVER_HANS
"""
struct FlatRule <: AbstractLRPRule end
modify_weight!(::FlatRule, w) = fill!(w, 1)
modify_bias!(::FlatRule, b) = fill!(b, 0)
modify_input(::FlatRule, input) = ones_like(input)

"""
    PassRule()

Pass-through rule. Passes relevance through to the lower layer.

Supports layers with constant input and output shapes, e.g. reshaping layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = R_j^{k+1}
```
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

LRP-``zᴮ``-rule. Commonly used on the first layer for pixel input.

The parameters `low` and `high` should be set to the lower and upper bounds
of the input features, e.g. `0.0` and `1.0` for raw image data.
It is also possible to provide two arrays of that match the input size.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k=\\sum_i \\frac{w_{ij}a_j^k - w_{ij}^{+}l_j - w_{ij}^{-}h_j}
    {\\sum_l w_{il}a_l^k+b_i - \\left(w_{il}^{+}l_l+b_i^{+}\\right) - \\left(w_{il}^{-}h_l+b_i^{-}\\right)} R_i^{k+1}
```

# References
- $REF_MONTAVON_OVERVIEW
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
    zₖ₊₁, pullback = Zygote.pullback(preactivation(layer), aₖ)

    # Compute pullback for W⁺, b⁺
    modify_layer!(Val(:keep_positive), layer)
    zₖ₊₁⁺, pullback⁺ = Zygote.pullback(preactivation(layer), l)
    reset!()

    # Compute pullback for W⁻, b⁻
    modify_layer!(Val(:keep_negative), layer)
    zₖ₊₁⁻, pullback⁻ = Zygote.pullback(preactivation(layer), h)

    # Evaluate pullbacks
    sₖ₊₁ = Rₖ₊₁ ./ modify_denominator(rule, zₖ₊₁ - zₖ₊₁⁺ - zₖ₊₁⁻)
    Rₖ .= -h .* only(pullback⁻(sₖ₊₁))
    reset!()  # re-modify mutated pullback
    Rₖ .+= aₖ .* only(pullback(sₖ₊₁))
    modify_layer!(Val(:keep_positive), layer)  # re-modify mutated pullback
    Rₖ .-= l .* only(pullback⁺(sₖ₊₁))
    reset!()
    return nothing
end

zbox_input(in::AbstractArray{T}, c::Real) where {T} = fill(convert(T, c), size(in))
function zbox_input(in::AbstractArray{T}, A::AbstractArray) where {T}
    @assert size(A) == size(in)
    return convert.(T, A)
end

"""
    AlphaBetaRule([α=2.0], [β=1.0])

LRP-``αβ`` rule. Weights positive and negative contributions according to the
parameters `α` and `β` respectively. The difference `α-β` must be equal to one.
Commonly used on lower layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\left(
    \\alpha\\frac{\\left(w_{ij}a_j^k\\right)^+}{\\sum_l\\left(w_{il}a_l^k+b_i\\right)^+}
    -\\beta\\frac{\\left(w_{ij}a_j^k\\right)^-}{\\sum_l\\left(w_{il}a_l^k+b_i\\right)^-}
\\right) R_i^{k+1}
```

# Optional arguments
- `α`: Multiplier for the positive output term, defaults to `2.0`.
- `β`: Multiplier for the negative output term, defaults to `1.0`.

# References
- $REF_BACH_LRP
- $REF_MONTAVON_OVERVIEW
"""
struct AlphaBetaRule{T} <: AbstractLRPRule
    α::T
    β::T
    function AlphaBetaRule(alpha=2.0f0, beta=1.0f0)
        alpha < 0 && throw(ArgumentError("Parameter `alpha` must be ≥0."))
        beta < 0 && throw(ArgumentError("Parameter `beta` must be ≥0."))
        !isone(alpha - beta) && throw(ArgumentError("`alpha - beta` must be equal one."))
        return new{eltype(alpha)}(alpha, beta)
    end
end

# The AlphaBetaRule requires its own implementation of relevance propagation.
function lrp!(Rₖ, rule::AlphaBetaRule, layer::L, aₖ, Rₖ₊₁) where {L}
    require_weight_and_bias(rule, layer)
    reset! = get_layer_resetter(rule, layer)

    aₖ⁺ = keep_positive(aₖ)
    aₖ⁻ = keep_negative(aₖ)

    # α: positive contributions
    modify_layer!(Val(:keep_negative_zero_bias), layer)
    zₖ₊₁ᵅ⁻, pullbackᵅ⁻ = Zygote.pullback(preactivation(layer), aₖ⁻)
    reset!()
    modify_layer!(Val(:keep_positive), layer)
    zₖ₊₁ᵅ⁺, pullbackᵅ⁺ = Zygote.pullback(preactivation(layer), aₖ⁺)
    # evaluate pullbacks
    sₖ₊₁ᵅ = Rₖ₊₁ ./ modify_denominator(rule, zₖ₊₁ᵅ⁺ + zₖ₊₁ᵅ⁻)
    Rₖ .= rule.α .* aₖ⁺ .* only(pullbackᵅ⁺(sₖ₊₁ᵅ))
    reset!()
    modify_layer!(Val(:keep_negative_zero_bias), layer) # re-modify mutated pullback
    Rₖ .+= rule.α .* aₖ⁻ .* only(pullbackᵅ⁻(sₖ₊₁ᵅ))
    reset!()

    # β: Negative contributions
    modify_layer!(Val(:keep_positive_zero_bias), layer)
    zₖ₊₁ᵝ⁻, pullbackᵝ⁻ = Zygote.pullback(preactivation(layer), aₖ⁻) #
    reset!()
    modify_layer!(Val(:keep_negative), layer)
    zₖ₊₁ᵝ⁺, pullbackᵝ⁺ = Zygote.pullback(preactivation(layer), aₖ⁺)
    # evaluate pullbacks
    sₖ₊₁ᵝ = Rₖ₊₁ ./ modify_denominator(rule, zₖ₊₁ᵝ⁺ + zₖ₊₁ᵝ⁻)
    Rₖ .-= rule.β .* aₖ⁺ .* only(pullbackᵝ⁺(sₖ₊₁ᵝ))
    reset!()
    modify_layer!(Val(:keep_positive_zero_bias), layer)  # re-modify mutated pullback
    Rₖ .-= rule.β .* aₖ⁻ .* only(pullbackᵝ⁻(sₖ₊₁ᵝ))
    reset!()
    return nothing
end

"""
    ZPlusRule()

LRP-``z⁺`` rule. Commonly used on lower layers.

Equivalent to `AlphaBetaRule(1.0f0, 0.0f0)`, but slightly faster.
See also [`AlphaBetaRule`](@ref).

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{\\left(w_{ij}a_j^k\\right)^+}{\\sum_l\\left(w_{il}a_l^k+b_i\\right)^+} R_i^{k+1}
```

# References
- $REF_BACH_LRP
- $REF_MONTAVON_DTD
"""
struct ZPlusRule <: AbstractLRPRule end
function lrp!(Rₖ, rule::ZPlusRule, layer::L, aₖ, Rₖ₊₁) where {L}
    require_weight_and_bias(rule, layer)
    reset! = get_layer_resetter(rule, layer)

    aₖ⁺ = keep_positive(aₖ)
    aₖ⁻ = keep_negative(aₖ)

    # Linearize around positive & negative activations (aₖ⁺, aₖ⁻)
    modify_layer!(Val(:keep_positive), layer)
    zₖ₊₁⁺, pullback⁺ = Zygote.pullback(layer, aₖ⁺)
    reset!()
    modify_layer!(Val(:keep_negative_zero_bias), layer)
    zₖ₊₁⁻, pullback⁻ = Zygote.pullback(layer, aₖ⁻)

    # Evaluate pullbacks
    sₖ₊₁ = Rₖ₊₁ ./ modify_denominator(rule, zₖ₊₁⁺ + zₖ₊₁⁻)
    Rₖ .= aₖ⁻ .* only(pullback⁻(sₖ₊₁))
    reset!()
    modify_layer!(Val(:keep_positive), layer) # re-modify mutated pullback
    Rₖ .+= aₖ⁺ .* only(pullback⁺(sₖ₊₁))
    reset!()
    return nothing
end

# Special cases for rules that don't modify params for extra performance:
for R in (ZeroRule, EpsilonRule)
    for L in (DropoutLayer, ReshapingLayer)
        @eval lrp!(Rₖ, ::$R, l::$L, aₖ, Rₖ₊₁) = lrp!(Rₖ, PassRule(), l, aₖ, Rₖ₊₁)
    end
end

# Fast implementation for Dense layer using Tullio.jl's einsum notation:
for R in (ZeroRule, EpsilonRule, GammaRule)
    @eval function lrp!(Rₖ, rule::$R, layer::Dense, aₖ, Rₖ₊₁)
        reset! = get_layer_resetter(rule, layer)
        modify_layer!(rule, layer)
        ãₖ = modify_input(rule, aₖ)
        zₖ₊₁ = modify_denominator(rule, preactivation(layer, ãₖ))
        @tullio Rₖ[j, b] = layer.weight[i, j] * ãₖ[j, b] / zₖ₊₁[i, b] * Rₖ₊₁[i, b]
        reset!()
        return nothing
    end
end
function lrp!(Rₖ, ::FlatRule, layer::Dense, aₖ, Rₖ₊₁)
    n = size(Rₖ, 1) # number of input neurons connected to each output neuron
    for i in axes(Rₖ, 2) # samples in batch
        fill!(view(Rₖ, :, i), sum(view(Rₖ₊₁, :, i)) / n)
    end
    return nothing
end
function lrp!(Rₖ, ::WSquareRule, layer::Dense, aₖ, Rₖ₊₁)
    den = sum(layer.weight .^ 2; dims=2)
    @tullio Rₖ[j, b] = layer.weight[i, j]^2 / den[i] * Rₖ₊₁[i, b]
    return nothing
end
