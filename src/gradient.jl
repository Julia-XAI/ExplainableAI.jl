struct Gradient{C<:Chain} <: AbstractXAIMethod
    model::C
end
function (analyzer::Gradient)(input, ns::AbstractNeuronSelector)
    return gradient((in) -> analyzer.model(in)[neuron_selection], input)[1]
end

struct InputTimesGradient{C<:Chain} <: AbstractXAIMethod
    model::C
end
function (analyzer::InputTimesGradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_neuron = ns(output)
    expl = input .* gradient((in) -> analyzer.model(in)[output_neuron], input)[1]
    return
end
