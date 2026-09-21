# Sum of the model output activations at a fixed `selection`.
# This is the scalar function differentiated w.r.t. the input.
masked_model(input, model, selection) = sum(model(input)[selection])

# Compute the gradient of the selected output activation(s) w.r.t. the input,
# returning the gradient, the model output and the output selection.
# Zygote and Enzyme are specialized manually in package extensions (#186);
# all other backends use this DifferentiationInterface.jl fallback.
function gradient_wrt_input(
        model, input, selector::AbstractOutputSelector, backend::AbstractADType
    )
    output = model(input)
    selection = selector(output)
    # Writing into a buffer keeps the array type of the input:
    # some backends return gradients of other types, e.g. forward-mode Enzyme.
    grad = DI.gradient!(
        masked_model,      # function differentiated w.r.t. its first argument
        similar(input),    # buffer the gradient is written into
        backend,           # AD backend
        input,             # active argument the gradient is taken w.r.t.
        DI.Constant(model),      # context argument held constant
        DI.Constant(selection),  # context argument held constant
    )
    return grad, output, selection
end

# Backend used to differentiate `masked_model` through DifferentiationInterface.jl,
# where the model is passed as an inactive context, not the differentiated function.
di_backend(backend::AbstractADType) = backend

# `AutoEnzyme`'s `function_annotation` says how to annotate the differentiated function.
# The split-mode extension differentiates the model directly, but here the model is an
# inactive context and the stateless `masked_model` is the function, which must not be
# annotated as differentiable, so the annotation is dropped.
di_backend(backend::AutoEnzyme) = AutoEnzyme(; mode = backend.mode)

# Fix the output selection ahead of time so that repeated gradients at the same
# `output_indices`, e.g. over augmented inputs, can reuse the preparation `prep`.
function prepare_gradient_wrt_input(model, input, output_indices, backend::AbstractADType)
    return DI.prepare_gradient(
        masked_model,      # function differentiated w.r.t. its first argument
        di_backend(backend),  # AD backend
        input,             # active argument the gradient is taken w.r.t.
        DI.Constant(model),          # context argument held constant
        DI.Constant(output_indices), # context argument held constant
    )
end

# Gradient at a fixed output selection, written into `grad` and reusing `prep`.
function gradient_wrt_input!(
        grad, model, input, output_indices, prep, backend::AbstractADType
    )
    return DI.gradient!(
        masked_model,      # function differentiated w.r.t. its first argument
        grad,              # buffer the gradient is written into
        prep,              # reused preparation
        di_backend(backend),  # AD backend
        input,             # active argument the gradient is taken w.r.t.
        DI.Constant(model),          # context argument held constant
        DI.Constant(output_indices), # context argument held constant
    )
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
    SmoothGrad(model)
    SmoothGrad(model, [n, std, rng])
    SmoothGrad(model, [n, distribution, rng])

Analyze model by calculating a smoothed sensitivity map.
This is done by averaging the gradient over `n` random samples
in a neighborhood of the input.
Defaults to 50 samples from the normal distribution with zero mean and `std=1.0f0`.

For optimal results, $REF_SMILKOV_SMOOTHGRAD recommends setting `std` between 10% and 20% of the input range of each sample,
e.g. `std = 0.1 * (maximum(input) - minimum(input))`.

## Keyword arguments
- `backend::AbstractADType`:
  AD backend used to compute gradients.
  Defaults to `$(DEFAULT_AD_BACKEND)`.

# References
- $REF_SMILKOV_SMOOTHGRAD
"""
struct SmoothGrad{M, B <: AbstractADType, D <: Sampleable, R <: AbstractRNG} <:
    AbstractXAIMethod
    model::M
    backend::B
    n::Int
    distribution::D
    rng::R
    show_progress::Bool

    function SmoothGrad(
            model::M, backend::B, n::Int, distribution::D, rng::R, show_progress::Bool
        ) where {M, B <: AbstractADType, D <: Sampleable, R <: AbstractRNG}
        n < 1 && throw(ArgumentError("Number of samples `n` needs to be larger than zero."))
        return new{M, B, D, R}(model, backend, n, distribution, rng, show_progress)
    end
end
function SmoothGrad(
        model, n::Int = 50, distribution::Sampleable = Normal(0.0f0, 1.0f0),
        rng::AbstractRNG = GLOBAL_RNG, show_progress::Bool = true;
        backend::AbstractADType = DEFAULT_AD_BACKEND,
    )
    return SmoothGrad(model, backend, n, distribution, rng, show_progress)
end
function SmoothGrad(
        model, n::Int, std::Real, rng::AbstractRNG = GLOBAL_RNG, show_progress::Bool = true;
        backend::AbstractADType = DEFAULT_AD_BACKEND,
    )
    return SmoothGrad(model, n, Normal(zero(std), std^2), rng, show_progress; backend)
end

function call_analyzer(input, analyzer::SmoothGrad, ns::AbstractOutputSelector; kwargs...)
    # One forward pass on the unaugmented input fixes the output selection,
    # which is then reused for every noisy sample.
    output = analyzer.model(input)
    output_indices = ns(output)

    prep = prepare_gradient_wrt_input(
        analyzer.model, input, output_indices, analyzer.backend
    )
    grad = similar(input)
    sum_grad = zero(input)
    noisy_input = similar(input)

    p = Progress(analyzer.n; desc = "Sampling SmoothGrad...", enabled = analyzer.show_progress)
    for _ in 1:(analyzer.n)
        sample_noise!(noisy_input, input, analyzer.rng, analyzer.distribution)
        gradient_wrt_input!(
            grad, analyzer.model, noisy_input, output_indices, prep, analyzer.backend
        )
        sum_grad .+= grad
        next!(p)
    end

    val = sum_grad ./ analyzer.n
    return Explanation(val, input, output, output_indices, :SmoothGrad, :sensitivity, nothing)
end

"""
    IntegratedGradients(model, [n=50])

Analyze model by using the Integrated Gradients method.

## Keyword arguments
- `backend::AbstractADType`:
  AD backend used to compute gradients.
  Defaults to `$(DEFAULT_AD_BACKEND)`.

# References
- $REF_SUNDARARAJAN_AXIOMATIC
"""
struct IntegratedGradients{M, B <: AbstractADType} <: AbstractXAIMethod
    model::M
    backend::B
    n::Int

    function IntegratedGradients(
            model::M, backend::B, n::Int
        ) where {M, B <: AbstractADType}
        n < 2 && throw(
            ArgumentError("Number of interpolation steps `n` needs to be larger than one."),
        )
        return new{M, B}(model, backend, n)
    end
end
function IntegratedGradients(model, n::Int = 50; backend::AbstractADType = DEFAULT_AD_BACKEND)
    return IntegratedGradients(model, backend, n)
end

function call_analyzer(
        input, analyzer::IntegratedGradients, ns::AbstractOutputSelector; input_ref = zero(input)
    )
    size(input) != size(input_ref) &&
        throw(ArgumentError("Input reference size doesn't match input size."))

    # The input is the endpoint α = 1 of the interpolation path.
    # Its gradient computation also provides the model output and the output selection.
    grad, output, output_indices = gradient_wrt_input(
        analyzer.model, input, ns, analyzer.backend
    )
    prep = prepare_gradient_wrt_input(
        analyzer.model, input, output_indices, analyzer.backend
    )

    # Integrate the gradient along the straight path xᵣ + α (x - xᵣ) for α ∈ [0, 1],
    # using the trapezoidal rule on `n` equidistant points, endpoints included.
    # Every point is computed from the endpoints, so `input_ref` is never mutated.
    T = eltype(input)
    input_delta = input - input_ref
    input_aug = similar(input)
    grad_buffer = similar(input)

    # Endpoints α = 1 (the input) and α = 0 carry half weight
    sum_grad = T(0.5) .* grad
    input_aug .= input_ref
    gradient_wrt_input!(
        grad_buffer, analyzer.model, input_aug, output_indices, prep, analyzer.backend
    )
    sum_grad .+= T(0.5) .* grad_buffer

    # Interior points carry full weight
    for k in 1:(analyzer.n - 2)
        input_aug .= input_ref .+ T(k / (analyzer.n - 1)) .* input_delta
        gradient_wrt_input!(
            grad_buffer, analyzer.model, input_aug, output_indices, prep, analyzer.backend
        )
        sum_grad .+= grad_buffer
    end

    val = input_delta .* sum_grad ./ (analyzer.n - 1)
    return Explanation(
        val, input, output, output_indices, :IntegratedGradients, :sensitivity, nothing
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
backend(analyzer::SmoothGrad) = analyzer.backend
backend(analyzer::IntegratedGradients) = analyzer.backend
backend(aug::NoiseAugmentation) = backend(aug.analyzer)
backend(aug::InterpolationAugmentation) = backend(aug.analyzer)
