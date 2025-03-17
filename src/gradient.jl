function forward_with_output_selection(model, input, selector::AbstractOutputSelector)
    output = model(input)
    sel = selector(output)
    return output[sel]
end

function gradient_wrt_input(
    model, input, output_selector::AbstractOutputSelector, backend::AbstractADType
)
    output = model(input)
    return gradient_wrt_input(model, input, output, output_selector, backend)
end

function gradient_wrt_input(
    model, input, output, output_selector::AbstractOutputSelector, backend::AbstractADType
)
    output_selection = output_selector(output)
    dy = zero(output)
    dy[output_selection] .= 1

    output, pbs = value_and_pullback(model, backend, input, tuple(dy))
    grad = only(pbs)
    return grad, output, output_selection
end

"""
    Gradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
"""
struct Gradient{M,B<:AbstractADType} <: AbstractXAIMethod
    model::M
    backend::B

    function Gradient(model::M, backend::B=DEFAULT_AD_BACKEND) where {M,B<:AbstractADType}
        new{M,B}(model, backend)
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
struct InputTimesGradient{M,B<:AbstractADType} <: AbstractXAIMethod
    model::M
    backend::B

    function InputTimesGradient(
        model::M, backend::B=DEFAULT_AD_BACKEND
    ) where {M,B<:AbstractADType}
        new{M,B}(model, backend)
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
    SmoothGrad(analyzer)
    SmoothGrad(analyzer, [n, std, rng]])
    SmoothGrad(analyzer, [n, distribution, rng])

Analyze model by calculating a smoothed sensitivity map.
This is done by averaging sensitivity maps of a `Gradient` analyzer over random samples
in a neighborhood of the input.
Defaults to 50 samples from the normal distribution with zero mean and `std=1.0f0`.

For optimal results, $REF_SMILKOV_SMOOTHGRAD recommends setting `std` between 10% and 20% of the input range of each sample,
e.g. `std = 0.1 * (maximum(input) - minimum(input))`.

# References
- $REF_SMILKOV_SMOOTHGRAD
"""
SmoothGrad(model, n=50, args...) = NoiseAugmentation(Gradient(model), n, args...)

"""
    IntegratedGradients(analyzer, [n=50])
    IntegratedGradients(analyzer, [n=50])

Analyze model by using the Integrated Gradients method.

# References
- $REF_SUNDARARAJAN_AXIOMATIC
"""
IntegratedGradients(model, n=50) = InterpolationAugmentation(Gradient(model), n)
