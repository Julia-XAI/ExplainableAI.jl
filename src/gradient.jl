function gradient_wrt_input(model, input::T, output_neuron)::T where {T}
    return only(gradient((in) -> model(in)[output_neuron], input))
end

"""
    Gradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
"""
struct Gradient{C<:Chain} <: AbstractXAIMethod
    model::C
    Gradient(model::Chain) = new{typeof(model)}(Flux.testmode!(check_ouput_softmax(model)))
end
function (analyzer::Gradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_neuron = ns(output)
    grad = gradient_wrt_input(analyzer.model, input, output_neuron)
    return Explanation(grad, output, output_neuron, :Gradient, Nothing)
end

"""
    InputTimesGradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
This gradient is then multiplied element-wise with the input.
"""
struct InputTimesGradient{C<:Chain} <: AbstractXAIMethod
    model::C
    function InputTimesGradient(model::Chain)
        return new{typeof(model)}(Flux.testmode!(check_ouput_softmax(model)))
    end
end
function (analyzer::InputTimesGradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_neuron = ns(output)
    attr = input .* gradient_wrt_input(analyzer.model, input, output_neuron)
    return Explanation(attr, output, output_neuron, :InputTimesGradient, Nothing)
end
