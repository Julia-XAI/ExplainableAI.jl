using ExplainableAI
using Documenter
using Literate

LITERATE_DIR = joinpath(@__DIR__, "src/literate")
OUT_DIR = joinpath(@__DIR__, "src/generated")

# Use Literate.jl to generate docs and notebooks of examples
function convert_literate(dir_in, dir_out)
    for p in readdir(dir_in)
        path = joinpath(dir_in, p)

        if isdir(path)
            convert_literate(path, joinpath(dir_out, p))
        else # isfile
            Literate.markdown(path, dir_out; documenter=true) # Markdown for Documenter.jl
            Literate.notebook(path, dir_out) # .ipynb notebook
            Literate.script(path, dir_out) # .jl script
        end
    end
end
convert_literate(LITERATE_DIR, OUT_DIR)

makedocs(;
    modules=[ExplainableAI],
    authors="Adrian Hill",
    sitename="ExplainableAI.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true", assets=String[]),
    #! format: off
    pages=[
        "Home" => "index.md",
        "Getting started" => "generated/example.md",
        "General usage" => Any[
            "Heatmapping"          => "generated/heatmapping.md",
            "Input augmentations"  => "generated/augmentations.md",
        ],
        "LRP" => Any[
            "Basic usage"                   => "generated/lrp/basics.md",
            "Assigning rules to layers"     => "generated/lrp/composites.md",
            "Supporting new layer types"    => "generated/lrp/custom_layer.md",
            "Custom LRP rules"              => "generated/lrp/custom_rules.md",
            "Concept Relevance Propagation" => "generated/lrp/crp.md",
            "Developer documentation"       => "lrp/developer.md"
        ],
        "API Reference" => Any[
            "General" => "api.md",
            "LRP"     => "lrp/api.md"
        ],
    ],
    #! format: on
    linkcheck=true,
    linkcheck_ignore=[
        r"https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10",
        r"https://www.nature.com/articles/s42256-023-00711-8",
    ],
    checkdocs=:exports, # only check docstrings in API reference if they are exported
)

deploydocs(; repo="github.com/adrhill/ExplainableAI.jl")
