const SingleChannelImage = AbstractArray{<:Real,2}
abstract type AbstractActivationNormalizer end
abstract type AbstractColorReducer end

"""
    heatmap(expl; kwargs...)

Visualize explanation.
"""
function heatmap(
    expl::AbstractArray;
    cs::ColorScheme=ColorSchemes.bwr,
    normalizer::AbstractActivationNormalizer=MaxAbsNormalizer(),
    reducer::AbstractColorReducer=SumReducer(),
    nchannels::Int=3,
    permute::Bool=true,
)
    img = normalizer(reducer(drop_singleton_dims(expl), nchannels))
    permute && (img = permutedims(img))
    return get(cs, img)
end

# Normalize activations across pixels
struct MaxAbsNormalizer <: AbstractActivationNormalizer end
function (::MaxAbsNormalizer)(img::SingleChannelImage)
    absmax = maximum(abs, img)
    return img / (2 * absmax) .+ 0.5
end

struct RangeNormalizer <: AbstractActivationNormalizer end
function normalize(img::SingleChannelImage)
    min, max = extrema(img)
    return (img .- min) / (max - min)
end

# Reduces activation in a pixel with multiple color channels into a single activation
struct MaxAbsReducer <: AbstractColorReducer end
function (::MaxAbsReducer)(img, nchannels)
    nchannels == 1 && return img
    dim = find_color_channel(img, nchannels)
    return dropdims(maximum(abs, img; dims=dim); dims=dim)
end

struct SumReducer <: AbstractColorReducer end
function (::SumReducer)(img, nchannels)
    nchannels == 1 && return img
    dim = find_color_channel(img, nchannels)
    return dropdims(sum(img; dims=dim); dims=dim)
end

function find_color_channel(img, nchannels)
    colordims = findall(size(img) .== nchannels)
    if length(colordims) == 0
        throw(ArgumentError("No dimension with nchannels=$nchannels color channels found."))
    elseif length(colordims) > 1
        throw(ArgumentError("Several dimensions of length $nchannels found."))
    end
    return first(colordims)
end
