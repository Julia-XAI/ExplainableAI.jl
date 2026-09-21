# Compute the gradient of the selected output activation(s) w.r.t. the input.
# Returns the gradient, the model output and the output selection.
# The selection depends on the model output,
# which requires a forward pass ahead of the differentiation.
function gradient_wrt_input(
        model, input, selector::AbstractOutputSelector, backend::AbstractADType
    )
    output = model(input)
    selection = selector(output)
    grad = DI.gradient(SelectedOutput(model, selection), backend, input)
    return grad, output, selection
end

# Sum of the output activations at a fixed `selection`.
struct SelectedOutput{M, S}
    model::M
    selection::S
end
(f::SelectedOutput)(input) = sum(f.model(input)[f.selection])

# Zygote evaluates the differentiated function exactly once, on the unmodified input,
# and tolerates side effects.
# Output and selection can therefore be recorded during a single forward pass (#186).
# This doesn't hold for backends in general:
# forward-mode and finite-difference backends call the function on dual-valued
# or perturbed inputs, and Enzyme doesn't support the write to `forward_pass`.
function gradient_wrt_input(
        model, input, selector::AbstractOutputSelector, backend::AutoZygote
    )
    f = RecordedSelectedOutput(model, selector, Ref{Any}(nothing))
    grad = DI.gradient(f, backend, input)
    output, selection = f.forward_pass[]
    return grad, output, selection
end

# Sum of the output activations picked by `selector`,
# recording model output and selection in `forward_pass`.
struct RecordedSelectedOutput{M, S <: AbstractOutputSelector}
    model::M
    selector::S
    forward_pass::Base.RefValue{Any}
end
function (f::RecordedSelectedOutput)(input)
    output = f.model(input)
    selection = f.selector(output)
    f.forward_pass[] = (output, selection)
    return sum(output[selection])
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

function gradient_explanation(::Gradient, grad, input, output, output_indices)
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

function gradient_explanation(::InputTimesGradient, grad, input, output, output_indices)
    attr = input .* grad
    return Explanation(
        attr, input, output, output_indices, :InputTimesGradient, :attribution, nothing
    )
end

const GradientAnalyzer = Union{Gradient, InputTimesGradient}

function call_analyzer(
        input, analyzer::GradientAnalyzer, ns::AbstractOutputSelector; kwargs...
    )
    grad, output, output_indices = gradient_wrt_input(
        analyzer.model, input, ns, analyzer.backend
    )
    return gradient_explanation(analyzer, grad, input, output, output_indices)
end

# Input augmentations fix the output selection ahead of sampling.
# All samples therefore differentiate the same `SelectedOutput`,
# reusing a DifferentiationInterface.jl preparation and a gradient buffer.
struct PreparedGradient{F <: SelectedOutput, P, G, O}
    f::F
    prep::P
    grad::G
    output::O
end

function prepare_augmentation(analyzer::GradientAnalyzer, input, output, output_indices)
    f = SelectedOutput(analyzer.model, output_indices)
    prep = DI.prepare_gradient(f, analyzer.backend, input)
    return PreparedGradient(f, prep, similar(input), output)
end

# The returned explanation aliases the gradient buffer of `p`,
# which is overwritten by the next call.
# It holds the model output of the unaugmented input.
function explain_augmentation(analyzer::GradientAnalyzer, input, p::PreparedGradient)
    DI.gradient!(p.f, p.grad, p.prep, analyzer.backend, input)
    return gradient_explanation(analyzer, p.grad, input, p.output, p.f.selection)
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
