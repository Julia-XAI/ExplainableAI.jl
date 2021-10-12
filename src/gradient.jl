struct Gradient{C<:Chain} <: AbstractXAIMethod
    model::C
end
function (analyzer::Gradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_neuron = ns(output)
    expl = gradient((in) -> analyzer.model(in)[output_neuron], input)[1]
    return output, expl
end

struct InputTimesGradient{C<:Chain} <: AbstractXAIMethod
    model::C
end
function (analyzer::InputTimesGradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_neuron = ns(output)
    expl = input .* gradient((in) -> analyzer.model(in)[output_neuron], input)[1]
    return output, expl
end
