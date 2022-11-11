# https://adrhill.github.io/ExplainableAI.jl/stable/generated/advanced_lrp/#How-it-works-internally
abstract type AbstractLRPRule end

# Bibliography
const REF_BACH_LRP = "S. Bach et al., *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*"
const REF_LAPUSCHKIN_CLEVER_HANS = "S. Lapuschkin et al., *Unmasking Clever Hans predictors and assessing what machines really learn*"
const REF_MONTAVON_DTD = "G. Montavon et al., *Explaining Nonlinear Classification Decisions with Deep Taylor Decomposition*"
const REF_MONTAVON_OVERVIEW = "G. Montavon et al., *Layer-Wise Relevance Propagation: An Overview*"

# Generic LRP rule. Used by all rules without custom implementations.
function lrp!(Rₖ, rule::AbstractLRPRule, modified_layer, aₖ, Rₖ₊₁)
    ãₖ = modify_input(rule, aₖ)
    z, back = Zygote.pullback(modified_layer, ãₖ)
    s = Rₖ₊₁ ./ modify_denominator(rule, z)
    c = only(back(s))
    Rₖ .= ãₖ .* c
end

#####################################
# Functions used to implement rules #
#####################################

# The function that follow define the default fallbacks used by LRP rules
# when calling the generic `lrp!` implementation above.
# Rule types are used to dispatch on rule-specific implementations.

# To implement a new rule, extend the following functions for your rule type:
# - modify_input
# - modify_denominator
# - modify_parameters OR (modify_weight and modify_bias) OR modify_layer
# - is_compatible

const LRP_LAYER_MODIFICATION_DIAGRAM = """
Use of a custom function `modify_layer` will overwrite functionality of `modify_parameters`,
`modify_weight` and `modify_bias` for the implemented combination of rule and layer types.
This is due to the fact that internally, `modify_weight` and `modify_bias` are called
by the default implementation of `modify_layer`.
`modify_weight` and `modify_bias` in turn call `modify_parameters` by default.

The default call structure looks as follows:
```
┌─────────────────────────────────────────┐
│              modify_layer               │
└─────────┬─────────────────────┬─────────┘
          │ calls               │ calls
┌─────────▼─────────┐ ┌─────────▼─────────┐
│   modify_weight   │ │    modify_bias    │
└─────────┬─────────┘ └─────────┬─────────┘
          │ calls               │ calls
┌─────────▼─────────┐ ┌─────────▼─────────┐
│ modify_parameters │ │ modify_parameters │
└───────────────────┘ └───────────────────┘
```
"""

"""
    modify_input(rule, input)

Modify input activation before computing relevance propagation.
"""
modify_input(rule, input) = input

"""
    modify_denominator(rule, d)

Modify denominator ``z`` for numerical stability on the forward pass.
"""
modify_denominator(rule, d) = stabilize_denom(d, 1.0f-9)

"""
    is_compatible(rule, layer)

Check compatibility of a LRP-Rule with layer type.
"""
is_compatible(rule, layer) = has_weight_and_bias(layer)

struct LRPCompatibilityError <: Exception
    rule::String
    layer::String
    LRPCompatibilityError(rule, layer) = new("$rule", "$layer")
end
function Base.showerror(io::IO, e::LRPCompatibilityError)
    return print(io, "LRP rule", e.rule, "isn't compatible with layer ", e.layer)
end

"""
    modify_parameters(rule, parameter)

Modify parameters before computing the relevance.

## Note
$LRP_LAYER_MODIFICATION_DIAGRAM
"""
modify_parameters(rule, param) = param

"""
    modify_weight(rule, weight)

Modify layer weights before computing the relevance.

## Note
$LRP_LAYER_MODIFICATION_DIAGRAM
"""
modify_weight(rule, w) = modify_parameters(rule, w)

"""
    modify_bias(rule, bias)

Modify layer bias before computing the relevance.

## Note
$LRP_LAYER_MODIFICATION_DIAGRAM
"""
modify_bias(rule, b) = modify_parameters(rule, b)

"""
    modify_layer(rule, layer)

Modify layer before computing the relevance.

## Note
$LRP_LAYER_MODIFICATION_DIAGRAM
"""
function modify_layer(rule, layer)
    !is_compatible(rule, layer) && throw(LRPCompatibilityError(rule, layer))
    !has_weight_and_bias(layer) && return layer

    w = modify_weight(rule, layer.weight)
    layer.bias == false && (return copy_layer(layer, w, false))
    b = modify_bias(rule, layer.bias)
    return copy_layer(layer, w, b)
end

# Useful presets, used e.g. in AlphaBetaRule, ZBoxRule & ZPlusRule:
modify_parameters(::Val{:keep_positive}, p) = keep_positive(p)
modify_parameters(::Val{:keep_negative}, p) = keep_negative(p)

modify_weight(::Val{:keep_positive_zero_bias}, w) = keep_positive(w)
modify_bias(::Val{:keep_positive_zero_bias}, b) = zero(b)

modify_weight(::Val{:keep_negative_zero_bias}, w) = keep_negative(w)
modify_bias(::Val{:keep_negative_zero_bias}, b) = zero(b)

#############
# LRP Rules #
#############

# The following LRP rules use the generic `lrp!` implementation at the top of this file.

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
is_compatible(::ZeroRule, layer) = true # compatible with all layer types

"""
    EpsilonRule([epsilon=1.0f-6])

LRP-``ϵ`` rule. Commonly used on middle layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{w_{ij}a_j^k}{\\epsilon +\\sum_{l}w_{il}a_l^k+b_i} R_i^{k+1}
```

# Optional arguments
- `epsilon`: Optional stabilization parameter, defaults to `1f-6`.

# References
- $REF_BACH_LRP
"""
struct EpsilonRule{T} <: AbstractLRPRule
    ϵ::T
    EpsilonRule(epsilon=1.0f-6) = new{Float32}(epsilon)
end
modify_denominator(r::EpsilonRule, d) = stabilize_denom(d, r.ϵ)
is_compatible(::EpsilonRule, layer) = true # compatible with all layer types

"""
    GammaRule([gamma=0.25])

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
    GammaRule(gamma=0.25f0) = new{Float32}(gamma)
end
function modify_parameters(r::GammaRule, param::AbstractArray{T}) where {T}
    γ = convert(T, r.γ)
    return @. param + γ * relu(param)
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
modify_input(::WSquareRule, input) = ones_like(input)
modify_weight(::WSquareRule, w) = w .^ 2
modify_bias(::WSquareRule, b) = zero(b)

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
modify_input(::FlatRule, input) = ones_like(input)
modify_weight(::FlatRule, w) = ones_like(w)
modify_bias(::FlatRule, b) = zero(b)

#####################
# Complex LRP Rules #
#####################

# The following rules use custom `lrp!` implementations
# and optionally custom `modify_layer` functions which return multiple modified layers.
# The convention used here is to return multiple modified layers as named tuples.

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
end
# No extra checks as reshaping operation will throw an error if layer isn't compatible:
is_compatible(::PassRule, layer) = true

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
function modify_layer(::ZBoxRule, layer)
    return (
        layer  = layer,
        layer⁺ = modify_layer(Val(:keep_positive), layer),
        layer⁻ = modify_layer(Val(:keep_negative), layer),
    )
end

function lrp!(Rₖ, rule::ZBoxRule, modified_layers, aₖ, Rₖ₊₁)
    l = zbox_input(aₖ, rule.low)
    h = zbox_input(aₖ, rule.high)

    z, back = Zygote.pullback(modified_layers.layer, aₖ)
    z⁺, back⁺ = Zygote.pullback(modified_layers.layer⁺, l)
    z⁻, back⁻ = Zygote.pullback(modified_layers.layer⁻, h)

    s = Rₖ₊₁ ./ modify_denominator(rule, z - z⁺ - z⁻)
    c = only(back(s))
    c⁺ = only(back⁺(s))
    c⁻ = only(back⁻(s))
    @. Rₖ = aₖ * c - l * c⁺ - h * c⁻
end

zbox_input(in::AbstractArray{T}, c::Real) where {T} = fill(convert(T, c), size(in))
function zbox_input(in::AbstractArray{T}, A::AbstractArray) where {T}
    @assert size(A) == size(in)
    return convert.(T, A)
end

"""
    AlphaBetaRule([alpha=2.0], [beta=1.0])

LRP-``αβ`` rule. Weights positive and negative contributions according to the
parameters `alpha` and `beta` respectively. The difference ``α-β`` must be equal to one.
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
- `alpha`: Multiplier for the positive output term, defaults to `2.0`.
- `beta`: Multiplier for the negative output term, defaults to `1.0`.

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
function modify_layer(::AlphaBetaRule, layer)
    return (
        layerᵅ⁺ = modify_layer(Val(:keep_positive), layer),
        layerᵅ⁻ = modify_layer(Val(:keep_negative_zero_bias), layer),
        layerᵝ⁺ = modify_layer(Val(:keep_negative), layer),
        layerᵝ⁻ = modify_layer(Val(:keep_positive_zero_bias), layer),
    )
end

function lrp!(Rₖ, rule::AlphaBetaRule, modified_layers, aₖ, Rₖ₊₁)
    aₖ⁺ = keep_positive(aₖ)
    aₖ⁻ = keep_negative(aₖ)

    zᵅ⁺, backᵅ⁺ = Zygote.pullback(modified_layers.layerᵅ⁺, aₖ⁺)
    zᵅ⁻, backᵅ⁻ = Zygote.pullback(modified_layers.layerᵅ⁻, aₖ⁻)
    zᵝ⁺, backᵝ⁺ = Zygote.pullback(modified_layers.layerᵝ⁺, aₖ⁺)
    zᵝ⁻, backᵝ⁻ = Zygote.pullback(modified_layers.layerᵝ⁻, aₖ⁻)

    sᵅ = Rₖ₊₁ ./ modify_denominator(rule, zᵅ⁺ + zᵅ⁻)
    sᵝ = Rₖ₊₁ ./ modify_denominator(rule, zᵝ⁺ + zᵝ⁻)
    cᵅ⁺ = only(backᵅ⁺(sᵅ))
    cᵅ⁻ = only(backᵅ⁻(sᵅ))
    cᵝ⁺ = only(backᵝ⁺(sᵝ))
    cᵝ⁻ = only(backᵝ⁻(sᵝ))

    T = eltype(aₖ)
    α = convert(T, rule.α)
    β = convert(T, rule.β)
    @. Rₖ = α * (aₖ⁺ * cᵅ⁺ + aₖ⁻ * cᵅ⁻) - β * (aₖ⁺ * cᵝ⁺ + aₖ⁻ * cᵝ⁻)
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
function modify_layer(::ZPlusRule, layer)
    return (
        layer⁺ = modify_layer(Val(:keep_positive), layer),
        layer⁻ = modify_layer(Val(:keep_negative_zero_bias), layer),
    )
end

function lrp!(Rₖ, rule::ZPlusRule, modified_layers, aₖ, Rₖ₊₁)
    aₖ⁺ = keep_positive(aₖ)
    aₖ⁻ = keep_negative(aₖ)

    z⁺, back⁺ = Zygote.pullback(modified_layers.layer⁺, aₖ⁺)
    z⁻, back⁻ = Zygote.pullback(modified_layers.layer⁻, aₖ⁻)

    s = Rₖ₊₁ ./ modify_denominator(rule, z⁺ + z⁻)
    c⁺ = only(back⁺(s))
    c⁻ = only(back⁻(s))
    @. Rₖ = aₖ⁺ * c⁺ + aₖ⁻ * c⁻
end

###########################
# Perfomance improvements #
###########################

# The following functions aren't strictly necessary – tests still pass when removing them.
# However they improve performance on specific combinations of rule and layer types.

# Rules that don't require layer information:
for R in (ZeroRule, EpsilonRule)
    for L in (DropoutLayer, ReshapingLayer)
        @eval function lrp!(Rₖ, ::$R, l::$L, aₖ, Rₖ₊₁)
            return lrp!(Rₖ, PassRule(), l, aₖ, Rₖ₊₁)
        end
    end
end
function lrp!(Rₖ, ::FlatRule, ::Dense, aₖ, Rₖ₊₁)
    n = size(Rₖ, 1) # number of input neurons connected to each output neuron
    for i in axes(Rₖ, 2) # samples in batch
        fill!(view(Rₖ, :, i), sum(view(Rₖ₊₁, :, i)) / n)
    end
end

# Fast implementation for Dense layer using Tullio.jl's einsum notation:
for R in (ZeroRule, EpsilonRule, GammaRule)
    @eval function lrp!(Rₖ, rule::$R, modified_layer::Dense, aₖ, Rₖ₊₁)
        ãₖ = modify_input(rule, aₖ)
        z = modify_denominator(rule, modified_layer(ãₖ))
        @tullio Rₖ[j, b] = modified_layer.weight[i, j] * ãₖ[j, b] / z[i, b] * Rₖ₊₁[i, b]
    end
end
function lrp!(Rₖ, ::WSquareRule, modified_layer::Dense, aₖ, Rₖ₊₁)
    den = sum(modified_layer.weight; dims=2)
    @tullio Rₖ[j, b] = modified_layer.weight[i, j] / den[i] * Rₖ₊₁[i, b]
end
