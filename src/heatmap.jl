using Base: Symbol
"""
Visualize explanation.
"""
function heatmap(exp::AbstractArray, cs::ColorScheme=:bwr)
    expl = drop_singleton_dims(expl)
    expl .= absnormalize(expl)
    img = image_from_array(expl)
end

function drop_singleton_dims(A::AbstractArray)
    return dropdims(A; dims=tuple(findall(size(A) .== 1)...))
end

"""
Normalize all values in array between 0 and 1.
"""
function normalize(A::AbstractArray)
    min, max = extrema(A)
    return (A .- min) / (max - min)
end

"""
Normalize all values in array to [0, 1], mapping input of 0 to ouput of 0.5 .
"""
function absnormalize(A::AbstractArray)
    absmax = maximum(abs, A)
    return A / (2 * absmax) .+ 0.5
end

"""
Helper function around colorview that abstracts channel permutations.
"""
function image_from_array(
    ::Type{C}, expl::AbstractArray{T,3}; flipimg=false
) where {C<:Colorant,T<:Number}
    color_dims = findall(size(expl) .== length(C))
    if length(color_dims) != 1
        throw(ArgumentError("Several dimensions of length $(length(C)) found."))
    end
    color_dim = first(color_dims)

    # leftover dims assumed to be in order HW, can be flipped
    hw_dims = deleteat!([1, 2, 3], color_dim)
    flipimg && (reverse!(hw_dims))
    return colorview(C, permutedims(expl, tuple(color_dim, hw_dims...)))
end
function image_from_array(expl::AbstractArray{<:Number,3}; kwargs...)
    return image_from_array(RGB{N0f8}, expl; kwargs...)
end

function image_from_array(
    ::Type{C}, expl::AbstractArray{T,2}; flipimg=false
) where {C<:Colorant,T<:Number}
    if flipimg
        return colorview(C, permutedims(expl, (2, 1)))
    else
        return colorview(C, expl)
    end
end
function image_from_array(expl::AbstractArray{<:Number,2}; kwargs...)
    return image_from_array(Gray{N0f8}, expl; kwargs...)
end
