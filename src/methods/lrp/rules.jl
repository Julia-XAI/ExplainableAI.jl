abstract type AbstractReverseRule end

"""
Common layer types LRP rules work on.
"""
CommonLayer = Union{Dense,Conv,MaxPool}

"""
Constructs generic LRP rules, implemented according to [1, 2].

Arguments:
- `layer`: A Flux layer.
- `rulename`: The name of the rule used for printing the LRPChain summary
- `ρ`: Function that modifies weights and biases as proposed in [1].
- `add_ϵ`: Function that increments zₖ on the forward pass for stability [1].

Default kwargs correspond to the basic LRP-0 rule.

References:
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
[2] W. Samek et al., Explaining Deep Neural Networks and Beyond:
    A Review of Methods and Applications
"""
struct LRPRule{T,L,W,B} <: AbstractReverseRule
    layer::L # Flux layer
    ρW::W # weights after applying modifier ρ
    b::B # biases after applying modifier ρ
    add_ϵ::Function # increments zₖ on forward pass
    rulename::String

    function LRPRule(layer, rulename; ρ=identity, add_ϵ=identity, T=Float32)
        ρW = T.(ρ(layer.weight))

        if typeof(layer.bias) <: Flux.Zeros
            b = zeros(T, size(ρW, 1))
        else
            b = T.(layer.bias)
        end
        return new{T, typeof(layer),typeof(ρW),typeof(b)}(layer, ρW, b, add_ϵ, rulename)
    end
end

function (r::LRPRule{T,L,W,B})(
    aₖ::AbstractArray, # activations (forward)
    Rₖ₊₁::AbstractArray, # relevance scores (backward)
) where {T,L<:CommonLayer,W,B}
    # Forward pass
    function fwpass(a)
        z = r.add_ϵ(r.ρW * a + r.b)
        s = Rₖ₊₁ ./ (z .+ 1e-9)
        return z ⋅ s
    end
    c = gradient(fwpass, aₖ)[1]

    # Backward pass
    Rₖ = aₖ .* c
    return Rₖ
end

"""
Constructor for LRP-0 rule. Commonly used on upper layers.
"""
LRP_0(l; kwargs...) = LRPRule(l, "LRP-0"; kwargs...)

"""
Constructor for LRP-``ϵ`` rule. Commonly used on middle layers.

Arguments:
- `ϵ`: Optional stabilization parameter, defaults to `1f-6`.
"""
function LRP_ϵ(l; ϵ=1f-6, kwargs...)
    _add_ϵ(z) = (1 + ϵ * std(z)) * z
    return LRPRule(l, "LRP-ϵ, ϵ=$(ϵ)"; add_ϵ=_add_ϵ, kwargs...)
end

"""
Constructor for LRP-``γ`` rule. Commonly used on lower layers.

Arguments:
- `γ`: Optional multiplier for added positive weights, defaults to 0.25.
"""
function LRP_γ(l; γ=0.25, kwargs...)
    _ρ(w) = w + γ * relu.(w)
    return LRPRule(l, "LRP-γ, γ=$(γ)"; ρ=_ρ, kwargs...)
end

"""
Implements generic ``z^{\\mathcal{B}}``-rule., implemented according to [1, 2].

Arguments:
- `layer`: A Flux layer.
- `rulename`: The name of the rule used for printing the LRPChain summary
- `ρ`: Function that modifies weights and biases as proposed in [1].

Default kwargs correspond to the basic LRP-0 rule.

References:
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
[2] W. Samek et al., Explaining Deep Neural Networks and Beyond:
    A Review of Methods and Applications
"""
struct LRPZBoxRule{T,L,W,B} <: AbstractReverseRule
    layer::L # Flux layer
    W::W # weights
    W⁻::W
    W⁺::W
    b::B # biases
    b⁻::B
    b⁺::B
    rulename::String

    function LRPZBoxRule(layer, rulename; ρ=identity, T=Float32)
        W = T.(ρ(layer.weight))
        W⁻ = min.(0, W)
        W⁺ = max.(0, W)

        if typeof(layer.bias) <: Flux.Zeros
            b = zeros(T, size(ρW, 1))
        else
            b = T.(layer.bias)
        end
        b⁻ = min.(0, b)
        b⁺ = max.(0, b)

        return new{T, typeof(layer),typeof(W),typeof(b)}(layer, W, W⁻, W⁺, b, b⁻, b⁺, rulename)
    end
end

function (r::LRPZBoxRule{T,L,W,B})(
    aₖ::AbstractArray, Rₖ₊₁::AbstractArray
) where {T,L<:CommonLayer,W,B}
    onemat = ones(T, size(aₖ))
    l = onemat * minimum(aₖ)
    h = onemat * maximum(aₖ)

    # Forward pass
    function fwpass(a)
        f = r.W * a + r.b
        f⁺ = r.W⁺ * l + r.b⁺
        f⁻ = r.W⁻ * l + r.b⁻

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

"""
Constructor for LRP-`z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
LRP_zᴮ(l; kwargs...) = LRPZBoxRule(l, "LRP-zᴮ"; kwargs...)
