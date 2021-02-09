using LayerwiseRelevancePropagation
using Documenter

makedocs(;
    modules=[LayerwiseRelevancePropagation],
    authors="Adrian Hill",
    repo="https://github.com/adrhill/LayerwiseRelevancePropagation.jl/blob/{commit}{path}#L{line}",
    sitename="LayerwiseRelevancePropagation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://adrhill.github.io/LayerwiseRelevancePropagation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/adrhill/LayerwiseRelevancePropagation.jl",
)
