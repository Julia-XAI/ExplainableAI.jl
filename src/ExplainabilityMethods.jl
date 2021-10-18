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

# analyzers
export AbstractXAIMethod
export Gradient, InputTimesGradient
export LRP, LRPZero, LRPEpsilon, LRPGamma

# LRP rules
export AbstractLRPRule, LRPRuleset
export ZeroRule, EpsilonRule, GammaRule, ZBoxRule

# heatmapping
export heatmap

# utils
export strip_softmax

end # module
