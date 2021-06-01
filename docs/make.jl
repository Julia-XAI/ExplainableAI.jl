using ExplainabilityMethods
using Documenter

makedocs(;
    modules=[ExplainabilityMethods],
    authors="Adrian Hill",
    repo="https://github.com/adrhill/ExplainabilityMethods.jl/blob/{commit}{path}#L{line}",
    sitename="ExplainabilityMethods.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://adrhill.github.io/ExplainabilityMethods.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/adrhill/ExplainabilityMethods.jl")
