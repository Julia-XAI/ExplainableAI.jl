module ExplainableAI

using Reexport
@reexport using XAIBase

using Base.Iterators
using Distributions: Distribution, Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG
using Zygote
using Flux
using ImageCore
using ImageTransformations: imresize

include("compat.jl")
include("bibliography.jl")
include("utils.jl")
include("input_augmentation.jl")
include("gradient.jl")
include("preprocessing.jl")

export Gradient, InputTimesGradient
export NoiseAugmentation, SmoothGrad
export InterpolationAugmentation, IntegratedGradients
export preprocess_imagenet

end # module
