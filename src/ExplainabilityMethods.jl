module ExplainabilityMethods

using LinearAlgebra
using Flux
using Zygote
using ColorSchemes
using ImageCore
using Base.Iterators

include("analyze_api.jl")
include("flux.jl")
include("utils.jl")
include("neuron_selection.jl")
include("gradient.jl")
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
export ZeroRule, EpsilonRule, GammaRule, ZBoxRule
export modify_layer, modify_params, modify_denominator

# heatmapping
export heatmap

# utils
export strip_softmax, flatten_chain

end # module
