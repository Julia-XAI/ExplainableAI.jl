abstract type AbstractXAIMethod end

"""
    output_and_explain(input, analyzer)
    output_and_explain(input, analyzer, neuron_selection)

Return raw classifier output and explanation.
If `neuron_selection` is specified, the explanation will be calculated for that neuron.
Otherwise, the output neuron with the highest activation is automatically chosen.
"""
function output_and_explain(
    input::AbstractArray{<:Real},
    analyzer::AbstractXAIMethod,
    neuron_selection::Integer,
    kwargs...,
)
    return analyzer(input, IndexNS(neuron_selection); kwargs...)
end

function output_and_explain(
    input::AbstractArray{<:Real}, analyzer::AbstractXAIMethod; kwargs...
)
    return analyzer(input, MaxActivationNS(); kwargs...)
end

"""
    explain(input, analyzer)
    explain(input, analyzer, neuron_selection)

Return explanation for classifier output.
If `neuron_selection` is specified, the explanation will be calculated for that neuron.
Otherwise, the output neuron with the highest activation is automatically chosen.
"""
function explain(
    input::AbstractArray{<:Real}, analyzer::AbstractXAIMethod, args...; kwargs...
)
    _, expl = output_and_explain(input, analyzer, args...; kwargs...)
    return expl
end

"""
    classify_and_explain(input, analyzer)
    classify_and_explain(input, analyzer, neuron_selection)
    classify_and_explain(input, labels, analyzer)
    classify_and_explain(input, labels, analyzer, neuron_selection)

Return classification and explanation.
If no labels are provided, the index of the highest neuron activation will be returned.
If `neuron_selection` is specified, the explanation will be calculated for that neuron.
Otherwise, the output neuron with the highest activation is automatically chosen.
"""
function classify_and_explain(
    input::AbstractArray{<:Real},
    labels::AbstractVector{<:AbstractString},
    analyzer::AbstractXAIMethod,
    args...;
    kwargs...,
)
    output, expl = output_and_explain(input, analyzer, args...; kwargs...)
    label = Flux.onecold(output, labels)
    return label, expl
end

"""
Return index of output neuron with highest activation and explanation.
"""
function classify_and_explain(
    input::AbstractArray{<:Real}, analyzer::AbstractXAIMethod, args...; kwargs...
)
    output, expl = output_and_explain(input, analyzer, args...; kwargs...)
    return argmax(output), expl
end
