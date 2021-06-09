module ExplainabilityMethods

using LinearAlgebra
using Flux
using Statistics: std
using PrettyTables: pretty_table
using ColorSchemes

include("api/explain.jl")
include("methods/gradient.jl")
include("methods/lrp/rules.jl")
include("heatmap.jl")
include("utils.jl")

export explain, classify_and_explain

# analyzers
export Gradient

# rules
export LRP_0, LRP_γ, LRP_ϵ, LRP_zᴮ

# heatmapping
export heatmap
end # module
