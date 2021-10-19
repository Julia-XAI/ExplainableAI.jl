struct Gradient{C<:Chain} <: AbstractXAIMethod
    model::C
    Gradient(model::Chain) = new{typeof(model)}(Flux.testmode!(check_ouput_softmax(model)))
end
function (analyzer::Gradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_neuron = ns(output)
    expl = gradient((in) -> analyzer.model(in)[output_neuron], input)[1]
    return expl, output
end

struct InputTimesGradient{C<:Chain} <: AbstractXAIMethod
    model::C
    InputTimesGradient(model::Chain) = new{typeof(model)}(Flux.testmode!(check_ouput_softmax(model)))
end
function (analyzer::InputTimesGradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_neuron = ns(output)
    expl = input .* gradient((in) -> analyzer.model(in)[output_neuron], input)[1]
    return expl, output
end
