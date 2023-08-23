"""
    augment_batch_dim(input, n)

Repeat each sample in input batch n-times along batch dimension.
This turns arrays of size `(..., B)` into arrays of size `(..., B*n)`.

## Example
```julia-repl
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> augment_batch_dim(A, 3)
2×6 Matrix{Int64}:
 1  1  1  2  2  2
 3  3  3  4  4  4
```
"""
function augment_batch_dim(input::AbstractArray{T,N}, n) where {T,N}
    return repeat(input; inner=(ntuple(Returns(1), N - 1)..., n))
end

"""
    reduce_augmentation(augmented_input, n)

Reduce augmented input batch by averaging the explanation for each augmented sample.
"""
function reduce_augmentation(input::AbstractArray{T,N}, n) where {T<:AbstractFloat,N}
    # Allocate output array
    in_size = size(input)
    in_size[end] % n != 0 &&
        throw(ArgumentError("Can't reduce augmented batch size of $(in_size[end]) by $n"))
    out_size = (in_size[1:(end - 1)]..., div(in_size[end], n))
    out = similar(input, eltype(input), out_size)

    axs = axes(input, N)
    colons = ntuple(Returns(:), N - 1)
    for (i, ax) in enumerate(first(axs):n:last(axs))
        view(out, colons..., i) .=
            dropdims(sum(view(input, colons..., ax:(ax + n - 1)); dims=N); dims=N) / n
    end
    return out
end

"""
    augment_indices(indices, n)

Strip batch indices and return inidices for batch augmented by n samples.

## Example
```julia-repl
julia> inds = [CartesianIndex(5,1), CartesianIndex(3,2)]
2-element Vector{CartesianIndex{2}}:
 CartesianIndex(5, 1)
 CartesianIndex(3, 2)

julia> augment_indices(inds, 3)
6-element Vector{CartesianIndex{2}}:
 CartesianIndex(5, 1)
 CartesianIndex(5, 2)
 CartesianIndex(5, 3)
 CartesianIndex(3, 4)
 CartesianIndex(3, 5)
 CartesianIndex(3, 6)
```
"""
function augment_indices(inds::Vector{CartesianIndex{N}}, n) where {N}
    indices_wo_batch = [i.I[1:(end - 1)] for i in inds]
    return map(enumerate(repeat(indices_wo_batch; inner=n))) do (i, idx)
        CartesianIndex{N}(idx..., i)
    end
end

"""
    NoiseAugmentation(analyzer, n, [std=1, rng=GLOBAL_RNG])
    NoiseAugmentation(analyzer, n, distribution, [rng=GLOBAL_RNG])

A wrapper around analyzers that augments the input with `n` samples of additive noise sampled from `distribution`.
This input augmentation is then averaged to return an `Explanation`.
"""
struct NoiseAugmentation{A<:AbstractXAIMethod,D<:Sampleable,R<:AbstractRNG} <:
       AbstractXAIMethod
    analyzer::A
    n::Int
    distribution::D
    rng::R
end
function NoiseAugmentation(analyzer, n, distr::Sampleable, rng=GLOBAL_RNG)
    return NoiseAugmentation(analyzer, n, distr::Sampleable, rng)
end
function NoiseAugmentation(analyzer, n, σ::Real=0.1f0, args...)
    return NoiseAugmentation(analyzer, n, Normal(0.0f0, Float32(σ)^2), args...)
end

function (aug::NoiseAugmentation)(input, ns::AbstractNeuronSelector)
    # Regular forward pass of model
    output = aug.analyzer.model(input)
    output_indices = ns(output)

    # Call regular analyzer on augmented batch
    augmented_input = add_noise(augment_batch_dim(input, aug.n), aug.distribution, aug.rng)
    augmented_indices = augment_indices(output_indices, aug.n)
    augmented_expl = aug.analyzer(augmented_input, AugmentationSelector(augmented_indices))

    # Average explanation
    return Explanation(
        reduce_augmentation(augmented_expl.attribution, aug.n),
        output,
        output_indices,
        augmented_expl.analyzer,
        nothing,
    )
end

function add_noise(A::AbstractArray{T}, distr::Distribution, rng::AbstractRNG) where {T}
    return A + T.(rand(rng, distr, size(A)))
end

"""
    InterpolationAugmentation(model, [n=50])

A wrapper around analyzers that augments the input with `n` steps of linear interpolation
between the input and a reference input (typically `zero(input)`).
The gradients w.r.t. this augmented input are then averaged and multiplied with the
difference between the input and the reference input.
"""
struct InterpolationAugmentation{A<:AbstractXAIMethod} <: AbstractXAIMethod
    analyzer::A
    n::Int
end

function (aug::InterpolationAugmentation)(
    input, ns::AbstractNeuronSelector, input_ref=zero(input)
)
    size(input) != size(input_ref) &&
        throw(ArgumentError("Input reference size doesn't match input size."))

    # Regular forward pass of model
    output = aug.analyzer.model(input)
    output_indices = ns(output)

    # Call regular analyzer on augmented batch
    augmented_input = interpolate_batch(input, input_ref, aug.n)
    augmented_indices = augment_indices(output_indices, aug.n)
    augmented_expl = aug.analyzer(augmented_input, AugmentationSelector(augmented_indices))

    # Average gradients and compute explanation
    expl = (input - input_ref) .* reduce_augmentation(augmented_expl.attribution, aug.n)

    return Explanation(expl, output, output_indices, augmented_expl.analyzer, nothing)
end

"""
    interpolate_batch(x, x0, nsamples)

Augment batch along batch dimension using linear interpolation between input `x` and a reference input `x0`.

## Example
```julia-repl
julia> x = Float16.(reshape(1:4, 2, 2))
2×2 Matrix{Float16}:
 1.0  3.0
 2.0  4.0

julia> x0 = zero(x)
2×2 Matrix{Float16}:
 0.0  0.0
 0.0  0.0

julia> interpolate_batch(x, x0, 5)
2×10 Matrix{Float16}:
 0.0  0.25  0.5  0.75  1.0  0.0  0.75  1.5  2.25  3.0
 0.0  0.5   1.0  1.5   2.0  0.0  1.0   2.0  3.0   4.0
```
"""
function interpolate_batch(
    x::AbstractArray{T,N}, x0::AbstractArray{T,N}, nsamples
) where {T,N}
    in_size = size(x)
    outs = similar(x, (in_size[1:(end - 1)]..., in_size[end] * nsamples))
    colons = ntuple(Returns(:), N - 1)
    for (i, t) in enumerate(range(zero(T), oneunit(T); length=nsamples))
        outs[colons..., i:nsamples:end] .= x0 + t * (x - x0)
    end
    return outs
end
