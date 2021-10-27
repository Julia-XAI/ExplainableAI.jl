using ExplainabilityMethods
using Documenter
using Literate

EXAMPLE_DIR = joinpath(@__DIR__, "literate")
OUT_DIR = joinpath(@__DIR__, "src/generated")

# Use Literate.jl to generate docs and notebooks of examples
for example in readdir(EXAMPLE_DIR)
    EXAMPLE = joinpath(EXAMPLE_DIR, example)

    Literate.markdown(EXAMPLE, OUT_DIR; documenter=true) # markdown for Documenter.jl
    Literate.notebook(EXAMPLE, OUT_DIR) # .ipynb notebook
    Literate.script(EXAMPLE, OUT_DIR) # .jl script
end

DocMeta.setdocmeta!(
    ExplainabilityMethods, :DocTestSetup, :(using ExplainabilityMethods); recursive=true
)
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
    pages=[
        "Home" => "index.md",
        "Getting started" => "generated/example.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/adrhill/ExplainabilityMethods.jl")
