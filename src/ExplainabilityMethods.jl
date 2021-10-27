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

const ANALYZERS = Dict(
    "Gradient" => Gradient,
    "InputTimesGradient" => InputTimesGradient,
    "LRP" => LRP,
    "LRPZero" => LRPZero,
    "LRPEpsilon" => LRPEpsilon,
    "LRPGamma" => LRPGamma,
)

# LRP rules
export AbstractLRPRule
export ZeroRule, EpsilonRule, GammaRule, ZBoxRule
export modify_layer, modify_params, modify_denominator

const RULES = Dict(
    "ZeroRule" => ZeroRule,
    "EpsilonRule" => EpsilonRule,
    "GammaRule" => GammaRule,
    "ZBoxRule" => ZBoxRule,
)

# heatmapping
export heatmap

# utils
export strip_softmax, flatten_chain

end # module
