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
include("lrp/show.jl")
include("lrp/composite_presets.jl") # uses lrp/show.jl
include("heatmap.jl")
include("preprocessing.jl")
export analyze

# Analyzers
export AbstractXAIMethod, Explanation
export Gradient, InputTimesGradient
export NoiseAugmentation, SmoothGrad
export InterpolationAugmentation, IntegratedGradients
export LRP

# LRP rules
export AbstractLRPRule
export LRP_CONFIG
export ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule, PassRule
export ZBoxRule, ZPlusRule, AlphaBetaRule
export modify_input, modify_denominator
export modify_param!, modify_layer!
export check_compat

# LRP composites
export Composite, AbstractCompositePrimitive
export LayerRule, GlobalRule, RangeRule, FirstLayerRule, LastLayerRule
export GlobalTypeRule, RangeTypeRule, FirstLayerTypeRule, LastLayerTypeRule
export FirstNTypeRule, LastNTypeRule
# Default composites
export EpsilonGammaBox, EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat
export EpsilonAlpha2Beta1Flat
# Useful type unions
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer

# heatmapping
export heatmap

# utils
export strip_softmax, flatten_model, check_model, flatten_chain, canonize
export preprocess_imagenet
end # module
