module ExplainableAI

using Base.Iterators
using MacroTools: @forward
using Distributions: Distribution, Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG
using Flux
using Zygote
using Markdown

# Heatmapping:
using ImageCore
using ImageTransformations: imresize
using ColorSchemes

include("compat.jl")
include("bibliography.jl")
include("neuron_selection.jl")
include("analyze_api.jl")
include("flux_types.jl")
include("flux_layer_utils.jl")
include("flux_chain_utils.jl")
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
include("lrp/crp.jl")
include("heatmap.jl")
include("preprocessing.jl")
export analyze

# Analyzers
export AbstractXAIMethod, Explanation
export Gradient, InputTimesGradient
export NoiseAugmentation, SmoothGrad
export InterpolationAugmentation, IntegratedGradients
export LRP, CRP

# LRP rules
export AbstractLRPRule
export LRP_CONFIG
export ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule
export PassRule, ZBoxRule, ZPlusRule, AlphaBetaRule, GeneralizedGammaRule

# LRP composites
export Composite, AbstractCompositePrimitive
export ChainTuple, ParallelTuple
export LayerMap, GlobalMap, RangeMap, FirstLayerMap, LastLayerMap
export GlobalTypeMap, RangeTypeMap, FirstLayerTypeMap, LastLayerTypeMap
export FirstNTypeMap
export lrp_rules, show_layer_indices

# Default composites
export EpsilonGammaBox, EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat
export EpsilonAlpha2Beta1Flat
# Useful type unions
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer, NormalizationLayer

# heatmapping
export heatmap

# utils
export strip_softmax, flatten_model, canonize
export preprocess_imagenet

# Package extension backwards compatibility with Julia 1.6.
# For Julia 1.6, Tullio is treated as a normal dependency and always loaded.
# https://pkgdocs.julialang.org/v1/creating-packages/#Transition-from-normal-dependency-to-extension
if !isdefined(Base, :get_extension)
    include("../ext/TullioLRPRulesExt.jl")
end

end # module
