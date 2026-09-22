"""
    AugmentationSelector(index)

Neuron selector that passes through an augmented neuron selection.
"""
struct AugmentationSelector{I} <: AbstractOutputSelector
    indices::I
end
(s::AugmentationSelector)(out::AbstractMatrix) = s.indices

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
- `show_progress:Bool`: Show progress meter while sampling augmentations. Defaults to `true`.
"""
struct NoiseAugmentation{A <: AbstractXAIMethod, D <: Sampleable, R <: AbstractRNG} <:
    AbstractXAIMethod
    analyzer::A
    n::Int
    distribution::D
    rng::R
    show_progress::Bool

    function NoiseAugmentation(
            analyzer::A, n::Int, distribution::D, rng::R = GLOBAL_RNG, show_progress = true
        ) where {A <: AbstractXAIMethod, D <: Sampleable, R <: AbstractRNG}
        n < 1 && throw(ArgumentError("Number of samples `n` needs to be larger than zero."))
        return new{A, D, R}(analyzer, n, distribution, rng, show_progress)
    end
end
function NoiseAugmentation(
        analyzer, n::Int, std::T = 1.0f0, rng = GLOBAL_RNG, show_progress = true
    ) where {T <: Real}
    distribution = Normal(zero(T), std)
    return NoiseAugmentation(analyzer, n, distribution, rng, show_progress)
end

function call_analyzer(input, aug::NoiseAugmentation, ns::AbstractOutputSelector; kwargs...)
    # Select the output once on the unaugmented input, then hold it fixed
    # so every sample is analyzed at the same output neuron(s).
    output = aug.analyzer.model(input)
    output_indices = ns(output)
    output_selector = AugmentationSelector(output_indices)

    p = Progress(aug.n; desc = "Sampling NoiseAugmentation...", enabled = aug.show_progress)

    # First augmentation
    noisy_input = similar(input)
    sample_noise!(noisy_input, input, aug.rng, aug.distribution)
    expl_aug = aug.analyzer(noisy_input, output_selector)
    sum_val = copy(expl_aug.val)
    next!(p)

    # Further augmentations
    for _ in 2:(aug.n)
        sample_noise!(noisy_input, input, aug.rng, aug.distribution)
        expl_aug = aug.analyzer(noisy_input, output_selector)
        sum_val .+= expl_aug.val
        next!(p)
    end

    # Average explanation
    val = sum_val / aug.n

    return Explanation(
        val, input, output, output_indices, expl_aug.analyzer, expl_aug.heatmap, nothing
    )
end

# Fill `out` with additive noise sampled from `distribution` around `input`.
function sample_noise!(
        out::A, input::A, rng::AbstractRNG, distribution::Sampleable
    ) where {A <: AbstractArray}
    rand!(rng, distribution, out)
    out .+= input
    return out
end

"""
    InterpolationAugmentation(model, [n=50])

A wrapper around analyzers that augments the input with `n` points of linear interpolation
between a reference input (typically `zero(input)`) and the input, both endpoints included.
The explanations of these augmented inputs are integrated over the path
using the trapezoidal rule,
and multiplied with the difference between the input and the reference input.

The reference input can be set via the keyword argument `input_ref` of `analyze`.
"""
struct InterpolationAugmentation{A <: AbstractXAIMethod} <: AbstractXAIMethod
    analyzer::A
    n::Int

    function InterpolationAugmentation(analyzer::A, n::Int) where {A <: AbstractXAIMethod}
        n < 2 && throw(
            ArgumentError("Number of interpolation steps `n` needs to be larger than one."),
        )
        return new{A}(analyzer, n)
    end
end

function call_analyzer(
        input, aug::InterpolationAugmentation, ns::AbstractOutputSelector; input_ref = zero(input)
    )
    size(input) != size(input_ref) &&
        throw(ArgumentError("Input reference size doesn't match input size."))

    # The input is the endpoint α = 1 of the interpolation path.
    # Analyzing it also provides the model output and the output selection,
    # which are then held fixed across the remaining interpolation points.
    expl_input = aug.analyzer(input, ns)
    output = expl_input.output
    output_indices = expl_input.output_selection
    output_selector = AugmentationSelector(output_indices)

    # Integrate the analyzer along the straight path xᵣ + α (x - xᵣ) for α ∈ [0, 1],
    # using the trapezoidal rule on `n` equidistant points, endpoints included.
    # Every point is computed from the endpoints, so `input_ref` is never mutated.
    T = eltype(input)
    input_delta = input - input_ref
    input_aug = similar(input)
    function explain_at(α)
        input_aug .= input_ref .+ α .* input_delta
        return aug.analyzer(input_aug, output_selector)
    end

    # Endpoints α = 1 (the input) and α = 0 carry half weight
    sum_val = T(0.5) .* expl_input.val
    sum_val .+= T(0.5) .* explain_at(zero(T)).val

    # Interior points carry full weight
    for k in 1:(aug.n - 2)
        sum_val .+= explain_at(T(k / (aug.n - 1))).val
    end

    # Average gradients and compute explanation
    val = input_delta .* sum_val ./ (aug.n - 1)

    return Explanation(
        val, input, output, output_indices, expl_input.analyzer, expl_input.heatmap, nothing
    )
end
