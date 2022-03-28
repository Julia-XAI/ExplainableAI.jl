module ExplainableAI

using Base.Iterators
using LinearAlgebra
using Flux
using Zygote
using Tullio

# Heatmapping:
using ImageCore
using ColorSchemes

# Model checks:
using Markdown
using PrettyTables

include("neuron_selection.jl")
include("analyze_api.jl")
include("flux.jl")
include("utils.jl")
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
export modify_denominator, modify_params, modify_layer
export check_model

# heatmapping
export heatmap

# utils
export strip_softmax, flatten_model, flatten_chain

end # module
