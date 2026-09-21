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
    grad, output, output_indices = gradient_wrt_input(
        analyzer.model, input, ns, analyzer.backend
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
    grad, output, output_indices = gradient_wrt_input(
        analyzer.model, input, ns, analyzer.backend
    )
    attr = input .* grad
    return Explanation(
        attr, input, output, output_indices, :InputTimesGradient, :attribution, nothing
    )
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
