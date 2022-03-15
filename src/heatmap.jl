# NOTE: Heatmapping assumes Flux's WHCN convention (width, height, color channels, batch size).

const HEATMAPPING_PRESETS = Dict{Symbol,Tuple{ColorScheme,Symbol,Symbol}}(
    # Analyzer => (colorscheme, reduce, normalize)
    :LRP => (ColorSchemes.bwr, :sum, :centered),
    :InputTimesGradient => (ColorSchemes.bwr, :sum, :centered), # same as LRP
    :Gradient => (ColorSchemes.grays, :norm, :extrema),
)

"""
    heatmap(expl::Explanation; kwargs...)
    heatmap(attr::AbstractArray; kwargs...)

Visualize explanation.
Assumes the Flux's WHCN convention (width, height, color channels, batch size).

## Keyword arguments
-`cs::ColorScheme`: ColorScheme that is applied.
    When calling `heatmap` with an `Explanation`, the method default is selected.
    When calling `heatmap` with an array, the default is `ColorSchemes.bwr`.
-`reduce::Symbol`: How the color channels are reduced to a single number to apply a colorscheme.
    The following methods can be selected, which are then applied over the color channels
    for each "pixel" in the attribution:
    - `:sum`: sum up color channels
    - `:norm`: compute 2-norm over the color channels
    - `:maxabs`: compute `maximum(abs, x)` over the color channels in
    When calling `heatmap` with an `Explanation`, the method default is selected.
    When calling `heatmap` with an array, the default is `:sum`.
-`normalize::Symbol`: How the color channel reduced heatmap is normalized before the colorscheme is applied.
    Can be either `:extrema` or `:centered`.
    When calling `heatmap` with an `Explanation`, the method default is selected.
    When calling `heatmap` with an array, the default for use with the `bwr` colorscheme is `:centered`.
-`permute::Bool`: Whether to flip W&H input channels. Default is `true`.
"""

function heatmap(
    attr::AbstractArray;
    cs::ColorScheme=ColorSchemes.bwr,
    reduce::Symbol=:sum,
    normalize::Symbol=:centered,
    permute::Bool=true,
)
    _size = size(attr)
    length(_size) != 4 && throw(
        DomainError(
            _size,
            """heatmap assumes Flux's WHCN convention (width, height, color channels, batch size) for the input.
            Please reshape your attribution to match this format if your model doesn't adhere to this convention.""",
        ),
    )
    _size[end] != 1 && throw(
        DomainError(
            _size[end],
            """heatmap is only applicable to a single attribution, got a batch dimension of $(_size[end]).""",
        ),
    )

    img = _normalize(dropdims(_reduce(dropdims(attr; dims=4), reduce); dims=3), normalize)
    permute && (img = permutedims(img))
    return ColorSchemes.get(cs, img)
end
# Use HEATMAPPING_PRESETS for default kwargs when dispatching on Explanation
function heatmap(expl::Explanation; permute::Bool=true, kwargs...)
    _cs, _reduce, _normalize = HEATMAPPING_PRESETS[expl.analyzer]
    return heatmap(
        expl.attribution;
        reduce=get(kwargs, :reduce, _reduce),
        normalize=get(kwargs, :normalize, _normalize),
        cs=get(kwargs, :cs, _cs),
        permute=permute,
    )
end

# Normalize activations across pixels
function _normalize(attr, method::Symbol)
    if method == :centered
        min, max = (-1, 1) .* maximum(abs, attr)
    elseif method == :extrema
        min, max = extrema(attr)
    else
        throw(
            ArgumentError(
                "Color scheme normalizer :$method not supported, `normalize` should be :extrema or :centered",
            ),
        )
    end
    return (attr .- min) / (max - min)
end

# Reduce attributions across color channels into a single scalar – assumes WHCN convention
function _reduce(attr::T, method::Symbol) where {T}
    if size(attr, 3) == 1 # nothing need to reduce
        return attr
    elseif method == :maxabs
        return maximum(abs, attr; dims=3)
    elseif method == :norm
        return mapslices(norm, attr; dims=3)::T
    elseif method == :sum
        return sum(attr; dims=3)
    end
    throw(
        ArgumentError(
            "Color channel reducer :$method not supported, `reduce` should be :maxabs, :sum or :norm",
        ),
    )
end
