module ExplainabilityMethods

using LinearAlgebra
using Flux
using Statistics: std
using PrettyTables: pretty_table
using ColorSchemes
using CairoMakie
using ImageCore

include("api/explain.jl")
include("methods/gradient.jl")
include("methods/lrp/rules.jl")
include("visualization/heatmap.jl")
include("visualization/compare.jl")
include("utils.jl")

export explain, classify_and_explain

# analyzers
export Gradient, InputTimesGradient

# rules
export LRP_0, LRP_γ, LRP_ϵ, LRP_zᴮ

# heatmapping
export heatmap, compare

# utils
export model_summary

end # module
