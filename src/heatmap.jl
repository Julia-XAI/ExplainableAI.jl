"""
Visualize explanation.
"""
function heatmap(expl::AbstractArray; cs::ColorScheme=ColorSchemes.bwr, kwargs...)
    expl = reduce_color_channels(drop_singleton_dims(expl), kwargs...)
    expl .= absnormalize(expl)
    return get(cs, expl)
end

function drop_singleton_dims(A::AbstractArray)
    return dropdims(A; dims=tuple(findall(size(A) .== 1)...))
end

"""
Normalize all values in array to [0, 1], mapping input of 0 to ouput of 0.5 .
"""
function absnormalize(A::AbstractArray)
    absmax = maximum(abs, A)
    return A / (2 * absmax) .+ 0.5
end

"""
Normalize all values in array between 0 and 1.
"""
function normalize(A::AbstractArray)
    min, max = extrema(A)
    return (A .- min) / (max - min)
end

"""
Helper function around colorview that abstracts channel permutations.
"""
function reduce_color_channels(
    expl::AbstractArray{<:Number,3}; nchannels::Integer=3, reducer::String="absmax"
)
    color_dims = findall(size(expl) .== nchannels)
    if length(color_dims) != 1
        throw(ArgumentError("Several dimensions of length $(length(C)) found."))
    end
    color_dim = first(color_dims)

    if reducer == "absmax"
        return maximum(abs, expl; dims=color_dim)
    elseif reducer == "sum"
        return sum(expl; dims=color_dim)
    else
        throw(ArgumentError("Only reducers \"absmax\" and \"sum\" are currently implemented."))
    end
end
reduce_color_channels(expl::AbstractArray{<:Number,2}; kwargs...) = expl
