
"""
Generic LRP rule for dense layers.
Implemented according to "Layer-Wise Relevance Propagation: An Overview" chapter (10.2.2).

Default kwargs correspond to the basic LRP-0 rule.
"""
function LRP_generic(
    layer::Dense,
    a::AbstractVector, # activations (forward)
    R::AbstractVector; # relevance scores (backward)
    ρ::Function=identity,
    add_ϵ::Function=identity, # also sometimes called incr
)::AbstractVector
    ρW, ρb = ρ.(Flux.params(layer))

    # forward pass
    z = ρW * a + ρb
    s = R ./ (add_ϵ(z) .+ 1e-9)

    # println.(size.([ρW, ρb, a, R, z, s, ρW'ᵀ * s]))
    return R_prev = a .* (transpose(ρW) * s) # backward pass
end

"""
Generic LRP rule for convolutional and sum-pooling layers.
Implemented according to "Layer-Wise Relevance Propagation: An Overview" chapter (10.2.2).
According to (10.3.2), this can also be applied to max-pooling layers:

Default kwargs correspond to the basic LRP-0 rule.
"""
function LRP_generic(
    layer::Union{Conv,MeanPool,MaxPool},
    a::AbstractVector, # activations (forward)
    R::AbstractVector; # relevance scores (backward)
    ρ::Function=identity,
    add_ϵ::Function=identity, # also sometimes called incr
)::AbstractVector
    ρW, ρb = ρ.(Flux.params(layer))

    # forward pass
    z = ρW * a + ρb
    s = R ./ (add_ϵ(z) .+ 1e-9)

    # println.(size.([ρW, ρb, a, R, z, s, ρW'ᵀ * s]))
    return R_prev = a .* (transpose(ρW) * s) # backward pass
end

"""
Generic LRP rule for batch norm layers.
Implemented according to "Layer-Wise Relevance Propagation: An Overview" chapter (10.3.2):

> Batch Normalization Layers are commonly used to facilitate training and improve prediction accuracy. At test time, they simply consist of a centering and rescaling operation. These layers can therefore be absorbed by the adjacent linear layer without changing the function. This allows to recover the canonical neural network structure needed for applying LRP.
"""
function LRP_generic(
    layer::BatchNorm,
    a::AbstractVector, # activations (forward)
    R::AbstractVector; # relevance scores (backward)
    ρ::Function=identity,
    add_ϵ::Function=identity, # also sometimes called incr
)::AbstractVector
    return R_prev = R
end

"""
LRP-0 rule. Commonly used on upper layers.
"""
LRP_0(l, a, R) = LRP_generic(l, a, R)

"""
LRP-``ϵ`` rule. Commonly used on middle layers.
"""
function LRP_ϵ(l, a, R; ϵ=0.25)
    _add_ϵ(z) = (1 + ϵ * std(z)) * z
    return LRP_generic(l, a, R; add_ϵ=_add_ϵ)
end

"""
LRP-``γ`` rule. Commonly used on lower layers.
"""
function LRP_γ(l, a, R; γ=0.1)
    _ρ(w) = w + γ * relu(w)
    return LRP_generic(l, a, R; ρ=_ρ)
end

"""
Implements the ``z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for pixel input.
"""
function LRP_zB(
    layer::Dense,
    a::AbstractVector, # activations (forward)
    R::AbstractVector; # relevance scores (backward)
    ρ::Function=identity,
    l::Real=-1.0, # lower bound of pixel values
    h::Real=1.0,  # upper bound of pixel values
)::AbstractVector
    W, b = Flux.params(layer)
    W⁻, W⁺ = minmax.(0, W)

    h = fill(1.0, size(a))
    l = -h

    # forward pass
    z = W * a - W⁺ * l - W⁻ * h .+ 1e-9 # denominator of LRP rules
    s = R ./ z

    return R_prev =
        a .* (transpose(W) * s) - l .* (transpose(W⁺) * s) - h .* (transpose(W⁻) * s) # backward pass
end

"""
Implements the ``z^{\\mathcal{B}}``-rule.
Commonly used on the first layer for input in ``\\mathbb{R}^d``.
"""
# function LRP_zB(l::Dense, a, R; γ=0.1)
#     return # TODO
# end
