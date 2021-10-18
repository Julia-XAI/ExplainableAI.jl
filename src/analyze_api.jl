abstract type AbstractXAIMethod end
# All analyzers are implemented such that they return an explanation and the model output:
#   (method::AbstractXAIMethod)(input, ns::AbstractNeuronSelector) -> (expl, output)

"""
    analyze(input, method)
    analyze(input, method, neuron_selection)

Return raw classifier output and explanation.
If `neuron_selection` is specified, the explanation will be calculated for that neuron.
Otherwise, the output neuron with the highest activation is automatically chosen.
"""
function analyze(
    input::AbstractArray{<:Real},
    method::AbstractXAIMethod,
    neuron_selection::Integer,
    kwargs...,
)
    return method(input, IndexNS(neuron_selection); kwargs...)
end

function analyze(input::AbstractArray{<:Real}, method::AbstractXAIMethod; kwargs...)
    return method(input, MaxActivationNS(); kwargs...)
end
