using ExplainableAI
using Documenter
using Literate

LITERATE_DIR = joinpath(@__DIR__, "src/literate")
OUT_DIR = joinpath(@__DIR__, "src/generated")

# Use Literate.jl to generate docs and notebooks of examples
for file in readdir(LITERATE_DIR)
    path = joinpath(LITERATE_DIR, file)

    Literate.markdown(path, OUT_DIR; documenter=true) # markdown for Documenter.jl
    Literate.notebook(path, OUT_DIR) # .ipynb notebook
    Literate.script(path, OUT_DIR) # .jl script
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
