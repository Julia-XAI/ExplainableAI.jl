module ExplainableAI

using Reexport
@reexport using XAIBase
import XAIBase: call_analyzer

using Base.Iterators
using Distributions: Distribution, Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG

# Automatic differentiation
using ADTypes: AbstractADType, AutoZygote
using DifferentiationInterface: value_and_pullback
using Zygote
const DEFAULT_AD_BACKEND = AutoZygote()

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
