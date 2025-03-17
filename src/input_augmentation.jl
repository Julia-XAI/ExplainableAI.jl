"""
    AugmentationSelector(index)

Neuron selector that passes through an augmented neuron selection.
"""
struct AugmentationSelector{I} <: AbstractOutputSelector
    indices::I
end
(s::AugmentationSelector)(out) = s.indices

"""
    NoiseAugmentation(analyzer, n, [std::Real, rng])
    NoiseAugmentation(analyzer, n, [distribution::Sampleable, rng])

A wrapper around analyzers that augments the input with `n` samples of additive noise sampled from a scalar `distribution`.
This input augmentation is then averaged to return an `Explanation`.
Defaults to the normal distribution with zero mean and `std=1.0f0`.

For optimal results, $REF_SMILKOV_SMOOTHGRAD recommends setting `std` between 10% and 20% of the input range of each sample,
e.g. `std = 0.1 * (maximum(input) - minimum(input))`.

## Keyword arguments
- `rng::AbstractRNG`: Specify the random number generator that is used to sample noise from the `distribution`. 
  Defaults to `GLOBAL_RNG`. 
"""
struct NoiseAugmentation{A<:AbstractXAIMethod,D<:Sampleable,R<:AbstractRNG} <:
       AbstractXAIMethod
    analyzer::A
    n::Int
    distribution::D
    rng::R

    function NoiseAugmentation(
        analyzer::A, n::Int, distribution::D, rng::R=GLOBAL_RNG
    ) where {A<:AbstractXAIMethod,D<:Sampleable,R<:AbstractRNG}
        n < 2 &&
            throw(ArgumentError("Number of noise samples `n` needs to be larger than one."))
        return new{A,D,R}(analyzer, n, distribution, rng)
    end
end
function NoiseAugmentation(analyzer, n::Int, std::T=1.0f0, rng=GLOBAL_RNG) where {T<:Real}
    distribution = Normal(zero(T), std^2)
    return NoiseAugmentation(analyzer, n, distribution, rng)
end

function call_analyzer(input, aug::NoiseAugmentation, ns::AbstractOutputSelector; kwargs...)
    # Regular forward pass of model
    output = aug.analyzer.model(input)
    output_indices = ns(output)
    output_selector = AugmentationSelector(output_indices)

    # First augmentation
    input_aug = similar(input)
    input_aug = sample_noise!(input_aug, input, aug)
    expl_aug = aug.analyzer(input_aug, output_selector)
    sum_val = expl_aug.val

    # Further augmentations
    for _ in 2:(aug.n)
        input_aug = sample_noise!(input_aug, input, aug)
        expl_aug = aug.analyzer(input_aug, output_selector)
        sum_val += expl_aug.val
    end

    # Average explanation
    val = sum_val / aug.n

    return Explanation(
        val, input, output, output_indices, expl_aug.analyzer, expl_aug.heatmap, nothing
    )
end

function sample_noise!(
    out::A, input::A, aug::NoiseAugmentation
) where {T,A<:AbstractArray{T}}
    out .= input .+ rand(aug.rng, aug.distribution, size(input))
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

    function InterpolationAugmentation(analyzer::A, n::Int) where {A<:AbstractXAIMethod}
        n < 2 && throw(
            ArgumentError("Number of interpolation steps `n` needs to be larger than one."),
        )
        return new{A}(analyzer, n)
    end
end

function call_analyzer(
    input, aug::InterpolationAugmentation, ns::AbstractOutputSelector; input_ref=zero(input)
)
    size(input) != size(input_ref) &&
        throw(ArgumentError("Input reference size doesn't match input size."))

    # Regular forward pass of model
    output = aug.analyzer.model(input)
    output_indices = ns(output)
    output_selector = AugmentationSelector(output_indices)

    # First augmentations
    input_aug = input_ref
    expl_aug = aug.analyzer(input_aug, output_selector)
    sum_val = expl_aug.val

    # Further augmentations
    input_delta = (input - input_ref) / (aug.n - 1)
    for _ in 1:(aug.n)
        input_aug += input_delta
        expl_aug = aug.analyzer(input_aug, output_selector)
        sum_val += expl_aug.val
    end

    # Average gradients and compute explanation
    val = (input - input_ref) .* sum_val / aug.n

    return Explanation(
        val, input, output, output_indices, expl_aug.analyzer, expl_aug.heatmap, nothing
    )
end
