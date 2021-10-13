module ExplainabilityMethods

using LinearAlgebra
using Flux
using Zygote
using PrettyTables: pretty_table
using ColorSchemes
using CairoMakie
using ImageCore
using Base.Iterators

include("api.jl")
include("flux_utils.jl")
include("utils.jl")
include("neuron_selection.jl")
include("gradient.jl")
include("lrp_rules.jl")
include("lrp.jl")
include("visualization/heatmap.jl")
include("visualization/compare.jl")

export explain, classify_and_explain

# analyzers
export AbstractXAIMethod
export Gradient, InputTimesGradient
export LRP, LRPZero, LRPEpsilon, LRPGamma

# LRP rules
export AbstractLRPRule, LRPRuleset
export ZeroRule, EpsilonRule, GammaRule, ZBoxRule

# heatmapping
export heatmap, compare

# utils
export model_summary
export strip_softmax

end # module
