module ExplainabilityMethods

using LinearAlgebra
using Flux
using Zygote
using ColorSchemes
using ImageCore
using Base.Iterators
using Tullio

using Markdown
using PrettyTables

include("analyze_api.jl")
include("flux.jl")
include("utils.jl")
include("neuron_selection.jl")
include("gradient.jl")
include("lrp_checks.jl")
include("lrp_rules.jl")
include("lrp.jl")
include("heatmap.jl")

export analyze

# Analyzers
export AbstractXAIMethod
export Gradient, InputTimesGradient
export LRP, LRPZero, LRPEpsilon, LRPGamma

# LRP rules
export AbstractLRPRule
export LRP_CONFIG
export ZeroRule, EpsilonRule, GammaRule, ZBoxRule
export lrp, modify_params, modify_denominator
export check_model

# heatmapping
export heatmap

# utils
export strip_softmax, flatten_model, flatten_chain

end # module
