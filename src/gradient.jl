# Scalar-valued function that is differentiated to obtain the input gradient.
# It runs a single forward pass, selects the target output activation(s) and reduces
# them to a scalar. The forward-pass output is cached in the `output` field so that
# the caller can reuse it for the `Explanation` without a second forward pass (#186).
mutable struct SelectedModelOutput{M, S <: AbstractOutputSelector}
    model::M
    selector::S
    output::Any
end
SelectedModelOutput(model, selector) = SelectedModelOutput(model, selector, nothing)

function (f::SelectedModelOutput)(input)
    output = f.model(input)
    f.output = output
    selection = f.selector(output)
    return sum(output[selection])
end

# Prepare the input-gradient computation for repeated evaluation on inputs of matching
# type and size (e.g. the samples drawn by input augmentations). Returns the differentiated
# function and a DifferentiationInterface preparation object to be passed back below.
function prepare_gradient_wrt_input(
        model, input, selector::AbstractOutputSelector, backend::AbstractADType
    )
    f = SelectedModelOutput(model, selector)
    prep = prepare_gradient(f, backend, input)
    return (f, prep)
end

# Compute the gradient of the selected output activation(s) w.r.t. the input.
# A single forward pass determines the selection *and* the model output (#186),
# the latter being returned for use in the `Explanation`.
function gradient_wrt_input(
        model, input, selector::AbstractOutputSelector, backend::AbstractADType
    )
    f = SelectedModelOutput(model, selector)
    _, grad = value_and_gradient(f, backend, input)
    output = f.output
    return grad, output, selector(output)
end

# Variant reusing a preparation object created by `prepare_gradient_wrt_input`.
function gradient_wrt_input(
        model, input, ::AbstractOutputSelector, backend::AbstractADType, prep::Tuple
    )
    f, gradient_prep = prep
    _, grad = value_and_gradient(f, gradient_prep, backend, input)
    output = f.output
    return grad, output, f.selector(output)
end

# `prep === nothing` falls back to the unprepared, single-shot computation.
function gradient_wrt_input(
        model, input, selector::AbstractOutputSelector, backend::AbstractADType, ::Nothing
    )
    return gradient_wrt_input(model, input, selector, backend)
end

"""
    Gradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
"""
struct Gradient{M, B <: AbstractADType} <: AbstractXAIMethod
    model::M
    backend::B

    function Gradient(model::M, backend::B = DEFAULT_AD_BACKEND) where {M, B <: AbstractADType}
        return new{M, B}(model, backend)
    end
end

function call_analyzer(input, analyzer::Gradient, ns::AbstractOutputSelector; kwargs...)
    return gradient_explanation(analyzer, input, ns, nothing)
end

# Shared explanation builder, also used by input augmentations (via `augmented_explanation`)
# to reuse a preparation object `prep` across many samples. `prep === nothing` computes
# the gradient without preparation.
function gradient_explanation(analyzer::Gradient, input, ns::AbstractOutputSelector, prep)
    grad, output, output_indices = gradient_wrt_input(
        analyzer.model, input, ns, analyzer.backend, prep
    )
    return Explanation(
        grad, input, output, output_indices, :Gradient, :sensitivity, nothing
    )
end

"""
    InputTimesGradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
This gradient is then multiplied element-wise with the input.
"""
struct InputTimesGradient{M, B <: AbstractADType} <: AbstractXAIMethod
    model::M
    backend::B

    function InputTimesGradient(
            model::M, backend::B = DEFAULT_AD_BACKEND
        ) where {M, B <: AbstractADType}
        return new{M, B}(model, backend)
    end
end

function call_analyzer(
        input, analyzer::InputTimesGradient, ns::AbstractOutputSelector; kwargs...
    )
    return gradient_explanation(analyzer, input, ns, nothing)
end

function gradient_explanation(
        analyzer::InputTimesGradient, input, ns::AbstractOutputSelector, prep
    )
    grad, output, output_indices = gradient_wrt_input(
        analyzer.model, input, ns, analyzer.backend, prep
    )
    attr = input .* grad
    return Explanation(
        attr, input, output, output_indices, :InputTimesGradient, :attribution, nothing
    )
end

# Preparation interface used internally by input augmentations to amortize the cost of
# repeatedly differentiating the same model over many samples. `prep` is never exposed to
# users: `prepare_analyzer` builds it once and `augmented_explanation` threads it back into
# the gradient computation for each sample. Analyzers without a preparation opt out by
# returning `nothing`, in which case augmentations fall back to a plain analyzer call.
prepare_analyzer(::AbstractXAIMethod, input, ::AbstractOutputSelector) = nothing
function prepare_analyzer(
        analyzer::Union{Gradient, InputTimesGradient}, input, selector::AbstractOutputSelector
    )
    return prepare_gradient_wrt_input(analyzer.model, input, selector, analyzer.backend)
end

augmented_explanation(analyzer, input, selector, ::Nothing) = analyzer(input, selector)
function augmented_explanation(
        analyzer::Union{Gradient, InputTimesGradient}, input, selector, prep::Tuple
    )
    return gradient_explanation(analyzer, input, selector, prep)
end

"""
    backend(analyzer)

Return the automatic differentiation backend used by a gradient-based analyzer.

For analyzers that wrap another analyzer, 
such as [`NoiseAugmentation`](@ref) and [`InterpolationAugmentation`](@ref),
the backend of the wrapped analyzer is returned.

# Example
```julia-repl
julia> analyzer = SmoothGrad(model; backend = AutoEnzyme());

julia> backend(analyzer)
AutoEnzyme()
```
"""
backend(analyzer::Gradient) = analyzer.backend
backend(analyzer::InputTimesGradient) = analyzer.backend
backend(aug::NoiseAugmentation) = backend(aug.analyzer)
backend(aug::InterpolationAugmentation) = backend(aug.analyzer)

"""
    SmoothGrad(model)
    SmoothGrad(model, [n, std, rng])
    SmoothGrad(model, [n, distribution, rng])

Analyze model by calculating a smoothed sensitivity map.
This is done by averaging sensitivity maps of a `Gradient` analyzer over random samples
in a neighborhood of the input.
Defaults to 50 samples from the normal distribution with zero mean and `std=1.0f0`.

For optimal results, $REF_SMILKOV_SMOOTHGRAD recommends setting `std` between 10% and 20% of the input range of each sample,
e.g. `std = 0.1 * (maximum(input) - minimum(input))`.

## Keyword arguments
- `backend::AbstractADType`: 
  AD backend used by the internal [`Gradient`](@ref) analyzer. 
  Defaults to `$(DEFAULT_AD_BACKEND)`.

# References
- $REF_SMILKOV_SMOOTHGRAD
"""
function SmoothGrad(model, n = 50, args...; backend::AbstractADType = DEFAULT_AD_BACKEND)
    return NoiseAugmentation(Gradient(model, backend), n, args...)
end

"""
    IntegratedGradients(model, [n=50])

Analyze model by using the Integrated Gradients method.

## Keyword arguments
- `backend::AbstractADType`: 
  AD backend used by the internal [`Gradient`](@ref) analyzer.
  Defaults to `$(DEFAULT_AD_BACKEND)`.

# References
- $REF_SUNDARARAJAN_AXIOMATIC
"""
function IntegratedGradients(model, n = 50; backend::AbstractADType = DEFAULT_AD_BACKEND)
    return InterpolationAugmentation(Gradient(model, backend), n)
end
