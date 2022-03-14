# NOTE: Heatmapping assumes Flux's WHCN convention (width, height, color channels, batch size).

"""
    heatmap(expl; kwargs...)

Visualize explanation.
Assumes the Flux's WHCN convention (width, height, color channels, batch size).

## Keyword arguments
-`cs::ColorScheme`: ColorScheme that is applied. Defaults to `ColorSchemes.bwr`.
-`reduce::Symbol`: How the color channels are reduced to a single number to apply a colorscheme.
    Can be either `:sum` or `:maxabs`. `:sum` sums up all color channels for each pixel.
    `:maxabs` selects the `maximum(abs, x)` over the color channel in each pixel.
    Default is `:sum`.
-`normalize::Symbol`: How the color channel reduced heatmap is normalized before the colorscheme is applied.
    Can be either `:extrema` or `:centered`. Default for use with colorscheme `bwr` is `:centered`.
"""
function heatmap(
    expl::AbstractArray;
    cs::ColorScheme=ColorSchemes.bwr,
    reduce::Symbol=:sum,
    normalize::Symbol=:centered,
    permute::Bool=true,
)
    _size = size(expl)
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
    # drop batch dim -> reduce color channels -> normalize image -> apply color scheme
    img = _normalize(_reduce(dropdims(expl; dims=4), reduce), normalize)
    permute && (img = permutedims(img))
    return ColorSchemes.get(cs, img)
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

# Reduces activation in a pixel with multiple color channels into a single activation
function _reduce(attr, method::Symbol)
    if method == :maxabs
        return dropdims(maximum(abs, attr; dims=3); dims=3)
    elseif method == :sum
        return dropdims(sum(attr; dims=3); dims=3)
    end
    throw(
        ArgumentError(
            "Color channel reducer :$method not supported, `reduce` should be :maxabs or :sum",
        ),
    )
end
