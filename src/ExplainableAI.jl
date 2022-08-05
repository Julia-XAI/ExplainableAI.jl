module ExplainableAI

using Base.Iterators
using LinearAlgebra
using Distributions: Distribution, Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG
using Flux
using Zygote
using Tullio

# Heatmapping:
using ImageCore
using ImageTransformations: imresize
using ColorSchemes

# Model checks:
using Markdown
using PrettyTables

include("compat.jl")
include("neuron_selection.jl")
include("analyze_api.jl")
include("flux_types.jl")
include("flux_utils.jl")
include("utils.jl")
include("input_augmentation.jl")
include("gradient.jl")
include("lrp/canonize.jl")
include("lrp/checks.jl")
include("lrp/rules.jl")
include("lrp/composite.jl")
include("lrp/lrp.jl")
include("heatmap.jl")
include("preprocessing.jl")
export analyze

# Analyzers
export AbstractXAIMethod
export Gradient, InputTimesGradient
export NoiseAugmentation, SmoothGrad
export InterpolationAugmentation, IntegratedGradients
export LRP

# LRP rules
export AbstractLRPRule
export LRP_CONFIG
export ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule, PassRule
export ZBoxRule, AlphaBetaRule
export modify_input, modify_denominator
export modify_param!, modify_layer!
export check_model

# LRP composites
export Composite, AbstractCompositePrimitive
export LayerRule, FirstRule, LastRule, GlobalRule
export RuleMap, RangeRuleMap, FirstNRuleMap, LastNRuleMap
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer

# heatmapping
export heatmap

# utils
export strip_softmax, flatten_model, flatten_chain, canonize
export preprocess_imagenet
end # module
