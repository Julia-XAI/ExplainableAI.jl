abstract type AbstractXAIMethod end

function explain(
    input::AbstractArray,
    analyzer::AbstractXAIMethod,
    args...;
    kwargs...
)
    # Calling analyzers returns vals and explanations by default.
    _, expl = analyzer(input)
    return expl
end

function classify_and_explain(
    input::AbstractArray,
    analyzer::AbstractXAIMethod,
    args...;
    kwargs...
)
    return analyzer(input)
end
