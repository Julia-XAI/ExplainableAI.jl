abstract type AbstractXAIMethod end

"""
Return raw classifier output and explanation.
"""
function output_and_explain(
    input::AbstractArray{<:Real}, analyzer::AbstractXAIMethod, class::Integer, args...; kwargs...
)
    output = analyzer.model(input)
    expl = analyzer(input, output, class, args...; kwargs...)
    return output, expl
end

"""
Return raw classifier output and explanation for any output neuron `class`.
"""
function output_and_explain(
    input::AbstractArray{<:Real}, analyzer::AbstractXAIMethod, args...; kwargs...
)
    output = analyzer.model(input)
    expl = analyzer(input, output, argmax(output), args...; kwargs...)
    return output, expl
end

"""
Return explanation for neuron with highest activation.
"""
function explain(
    input::AbstractArray{<:Real}, analyzer::AbstractXAIMethod, args...; kwargs...
)
    _, expl = output_and_explain(input, analyzer, args...; kwargs...)
    return expl
end

"""
Return label of classification and explanation.
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
