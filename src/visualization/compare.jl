"""
Compare analyzers over one or more images.
"""
function compare(
    imgs::AbstractVector{<:AbstractArray},
    labels::AbstractVector{<:AbstractString},
    algs::AbstractVector{<:AbstractXAIMethod};
    backgroundcolor=:white, #:transparent
)
    n_algs = length(algs)
    n_imgs = length(imgs)
    fig = Figure(; backgroundcolor=backgroundcolor)

    # Add supertitle
    fig[0, :] = Label(fig, "ExplainabilityMethods.jl"; textsize=24, color=(:black, 0.6))

    for (row, img) in enumerate(imgs)
        # Run first analyzer to get class probs
        output, _ = output_and_explain(img, algs[1])
        lab = Flux.onecold(output, labels)
        println(lab)

        fig[row, 1, Left()] = Label(fig, "asd"; halign=:right, padding=(0, 5, 0, 0))
        ax = Axis(fig[row, 1]; aspect=1)
        raw = image!(ax, RGB.(img))
        hidedecorations!(ax) # remove tics

        for (col, alg) in enumerate(algs)
            ax = Axis(fig[row, col + 1]; aspect=1)

            attr = explain(img, alg)
            hm = heatmap!(ax, attr; colormap=:bwr)
            hm.colorrange = (-1, 1)
            hidedecorations!(ax)
        end
    end
    cbar = fig[1:n_imgs, n_algs + 2] = Colorbar(fig, hm; label="Activations")

    # Add title for Input
    alg_name_padding = (0, 0, 5, 0)
    alg_name_rotation = pi / 8
    fig[1, 1, Top()] = Label(
        fig,
        "Input";
        halign=:left,
        valign=:bottom,
        padding=alg_name_padding,
        rotation=alg_name_rotation,
    )

    # Add titles for Algorithms
    for (col, alg) in enumerate(algs)
        fig[1, col + 1, Top()] = Label(
            fig,
            string(typeof(alg));
            halign=:left,
            valign=:bottom,
            padding=alg_name_padding,
            rotation=alg_name_rotation,
        )
    end

    # Trim gaps
    rowgap!(fig.layout, 7)
    colgap!(fig.layout, 7)
    colgap!(fig.layout, n_algs + 1, 25) # larger space before colorbar
    trim!(fig.layout)

    return fig
end

function compare(
    img::AbstractArray,
    labels::AbstractVector{<:AbstractString},
    algs::AbstractVector{<:AbstractXAIMethod};
    kwargs...,
)
    return compare([img], labels, algs, kwargs...)
end
