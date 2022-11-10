using ExplainableAI
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

makedocs(;
    modules=[ExplainableAI],
    authors="Adrian Hill",
    sitename="ExplainableAI.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true", assets=String[]),
    pages=[
        "Home"            => "index.md",
        "Getting started" => "generated/example.md",
        "Advanced LRP"    => "generated/advanced_lrp.md",
        "API Reference"   => "api.md",
    ],
)

deploydocs(; repo="github.com/adrhill/ExplainableAI.jl")
