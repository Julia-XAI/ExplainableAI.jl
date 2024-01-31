module ExplainableAI

using Reexport
@reexport using XAIBase

using Base.Iterators
using Distributions: Distribution, Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG
using Zygote

include("compat.jl")
include("bibliography.jl")
include("input_augmentation.jl")
include("gradient.jl")
include("gradcam.jl")

export Gradient, InputTimesGradient
export NoiseAugmentation, SmoothGrad
export InterpolationAugmentation, IntegratedGradients
export GradCAM

end # module
