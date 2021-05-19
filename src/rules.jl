abstract type AbstractReverseRule end

"""
Common layer types LRP rules work on.
"""
CommonLayer = Union{Dense,Conv,MaxPool}

"""
Constructs generic LRP rules, implemented according to [1], chapter (10.2.2).

Arguments:
- `layer`: A Flux layer.
- `rulename`: The name of the rule used for printing the LRPChain summary
- `ρ`: Function that modifies weights and biases as proposed in [1].
- `add_ϵ`: Function that increments zₖ on the forward pass for stability [1].

Default kwargs correspond to the basic LRP-0 rule.

References:
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
"""
struct LRPRule{L,PW,PB} <: AbstractReverseRule
    layer::L # Flux layer
    ρW::PW # weights after applying modifier ρ
    ρb::PB # biases after applying modifier ρ
    add_ϵ::Function # increments zₖ on forward pass
    rulename::String

    function LRPRule(layer, rulename; ρ=identity, add_ϵ=identity)
        ρW, ρb = ρ.(Flux.params(layer))
        return new{typeof(layer),typeof(ρW),typeof(ρb)}(layer, ρW, ρb, add_ϵ, rulename)
    end
end

function (r::LRPRule{L,PW,PB})(
    aₖ::AbstractArray, # activations (forward)
    Rₖ₊₁::AbstractArray, # relevance scores (backward)
)::AbstractArray where {L<:CommonLayer, PW, PB}
    # forward pass

    function fwpass(a)
        z = r.add_ϵ(r.ρW * a + r.ρb)
        s = Rₖ₊₁ ./ (z .+ 1e-9)
        return z ⋅ s
    end

    c = gradient(fwpass, aₖ)
    println(c)

    Rₖ = aₖ .* c # backward pass
    return Rₖ
end

"""
LRP-0 rule. Commonly used on upper layers.
"""
LRP_0(l) = LRPRule(l, "LRP-0")

"""
LRP-``ϵ`` rule. Commonly used on middle layers.

Arguments:
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.
"""
function LRP_ϵ(l; ϵ=1f-6)
    _add_ϵ(z) = (1 + ϵ * std(z)) * z
    return LRPRule(l, "LRP-ϵ, ϵ=$(ϵ)"; add_ϵ=_add_ϵ)
end

"""
LRP-``γ`` rule. Commonly used on lower layers.

Arguments:
- `γ`: Optional multiplier for added positive weights.
"""
function LRP_γ(l; γ=0.25)
    _ρ(w) = w + γ * relu(w)
    return LRPRule(l, "LRP-γ, γ=$(γ)"; ρ=_ρ)
end

"""
Implements the ``z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
struct LRPZBoxRule{L,W,B} <: AbstractReverseRule
    layer::L # Flux layer
    W::W # weights
    W⁻::W
    W⁺::W
    b::B # biases
    rulename::String

    function LRPZBoxRule(layer; ρ=identity)
        rulename = "LRP-zᴮ"

        W, b = ρ.(Flux.params(layer))
        W⁻ = min.(0, W)
        W⁺ = max.(0, W)

        return new{typeof(layer),typeof(W),typeof(b)}(layer, W, W⁻, W⁺, b, rulename)
    end
end

function (r::LRPZBoxRule{L,W,B})(
    aₖ::AbstractArray, Rₖ₊₁::AbstractArray
)::AbstractArray where {L<:CommonLayer, W, B}
    l, h = extrema(aₖ)

    function fwpass(a)
        z = r.W * a - r.W⁺ * l - r.W⁻
        s = Rₖ₊₁ ./ (z .+ 1e-9)
        return z ⋅ s
    end

    dfw = gradient(fwpass, a)

    c = dfw(aₖ)
    cₗ= dfw(l)
    cₕ= dfw(h)

    println(c, cₗ, cₕ)

    # backward pass
    Rₖ = aₖ .* c + l .* cₗ + h .* cₕ
    return Rₖ
end
