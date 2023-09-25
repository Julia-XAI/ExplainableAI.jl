# NOTE: Heatmapping assumes Flux's WHCN convention (width, height, color channels, batch size).

const HEATMAPPING_PRESETS = Dict{Symbol,Tuple{ColorScheme,Symbol,Symbol}}(
    # Analyzer => (colorscheme, reduce, rangescale)
    :LRP => (ColorSchemes.seismic, :sum, :centered), # attribution
    :CRP => (ColorSchemes.seismic, :sum, :centered), # attribution
    :InputTimesGradient => (ColorSchemes.seismic, :sum, :centered), # attribution
    :Gradient => (ColorSchemes.grays, :norm, :extrema), # gradient
)

"""
    heatmap(explanation)
    heatmap(input, analyzer)
    heatmap(input, analyzer, neuron_selection)

Visualize explanation.
Assumes Flux's WHCN convention (width, height, color channels, batch size).

See also [`analyze`](@ref).

## Keyword arguments
- `cs::ColorScheme`: color scheme from ColorSchemes.jl that is applied.
    When calling `heatmap` with an `Explanation` or analyzer, the method default is selected.
    When calling `heatmap` with an array, the default is `ColorSchemes.seismic`.
- `reduce::Symbol`: selects how color channels are reduced to a single number to apply a color scheme.
    The following methods can be selected, which are then applied over the color channels
    for each "pixel" in the explanation:
    - `:sum`: sum up color channels
    - `:norm`: compute 2-norm over the color channels
    - `:maxabs`: compute `maximum(abs, x)` over the color channels
    When calling `heatmap` with an `Explanation` or analyzer, the method default is selected.
    When calling `heatmap` with an array, the default is `:sum`.
- `rangescale::Symbol`: selects how the color channel reduced heatmap is normalized
    before the color scheme is applied. Can be either `:extrema` or `:centered`.
    When calling `heatmap` with an `Explanation` or analyzer, the method default is selected.
    When calling `heatmap` with an array, the default for use with the `seismic` color scheme is `:centered`.
- `permute::Bool`: Whether to flip W&H input channels. Default is `true`.
- `unpack_singleton::Bool`: When heatmapping a batch with a single sample, setting `unpack_singleton=true`
    will return an image instead of an Vector containing a single image.
- `process_batch::Bool`: When heatmapping a batch, setting `process_batch=true`
    will apply the color channel reduction and normalization to the entire batch
    instead of computing it individually for each sample. Defaults to `false`.
"""
function heatmap(
    val::AbstractArray{T,N};
    cs::ColorScheme=ColorSchemes.seismic,
    reduce::Symbol=:sum,
    rangescale::Symbol=:centered,
    permute::Bool=true,
    unpack_singleton::Bool=true,
    process_batch::Bool=false,
) where {T,N}
    N != 4 && throw(
        ArgumentErrorError(
            "heatmap assumes Flux's WHCN convention (width, height, color channels, batch size) for the input.
            Please reshape your explanation to match this format if your model doesn't adhere to this convention.",
        ),
    )
    if unpack_singleton && size(val, 4) == 1
        return _heatmap(val[:, :, :, 1], cs, reduce, rangescale, permute)
    end
    if process_batch
        hs = _heatmap(val, cs, reduce, rangescale, permute)
        return eachslice(hs; dims=3)
    end
    return [_heatmap(v, cs, reduce, rangescale, permute) for v in eachslice(val; dims=4)]
end

# Use HEATMAPPING_PRESETS for default kwargs when dispatching on Explanation
function heatmap(expl::Explanation; kwargs...)
    _cs, _reduce, _rangescale = HEATMAPPING_PRESETS[expl.analyzer]
    return heatmap(
        expl.val;
        reduce=get(kwargs, :reduce, _reduce),
        rangescale=get(kwargs, :rangescale, _rangescale),
        cs=get(kwargs, :cs, _cs),
        kwargs...,
    )
end
# Analyze & heatmap in one go
function heatmap(input, analyzer::AbstractXAIMethod, args...; kwargs...)
    expl = analyze(input, analyzer, args...)
    return heatmap(expl; kwargs...)
end

# Lower level function that can be mapped along batch dimension
function _heatmap(val, cs::ColorScheme, reduce::Symbol, rangescale::Symbol, permute::Bool)
    img = dropdims(reduce_color_channel(val, reduce); dims=3)
    permute && (img = flip_wh(img))
    return ColorSchemes.get(cs, img, rangescale)
end

flip_wh(img::AbstractArray{T,2}) where {T} = permutedims(img, (2, 1))
flip_wh(img::AbstractArray{T,3}) where {T} = permutedims(img, (2, 1, 3))

# Reduce explanations across color channels into a single scalar – assumes WHCN convention
function reduce_color_channel(val::AbstractArray, method::Symbol)
    init = zero(eltype(val))
    if size(val, 3) == 1 # nothing to reduce
        return val
    elseif method == :sum
        return reduce(+, val; dims=3)
    elseif method == :maxabs
        return reduce((c...) -> maximum(abs.(c)), val; dims=3, init=init)
    elseif method == :norm
        return reduce((c...) -> sqrt(sum(c .^ 2)), val; dims=3, init=init)
    end

    throw(
        ArgumentError(
            "Color channel reducer :$method not supported, `reduce` should be :maxabs, :sum or :norm",
        ),
    )
end
