struct Gradient{C<:Chain} <: AbstractXAIMethod
    model::C
end

function (analyzer::Gradient)(input, _output, class)
    # Calculate gradient w.r.t. neuron with highest activation
    return gradient((in)-> analyzer.model(in)[class], input)[1]
end
