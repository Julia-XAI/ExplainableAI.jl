struct Gradient{C<:Chain} <: AbstractXAIMethod
    model::C
end
function (analyzer::Gradient)(input, _output, neuron_selection)
    return gradient((in) -> analyzer.model(in)[neuron_selection], input)[1]
end


struct InputTimesGradient{C<:Chain} <: AbstractXAIMethod
    model::C
end
function (analyzer::InputTimesGradient)(input, _output, neuron_selection)
    return input .* gradient((in) -> analyzer.model(in)[neuron_selection], input)[1]
end
