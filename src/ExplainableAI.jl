module ExplainableAI

using Reexport
@reexport using XAIBase
import XAIBase: call_analyzer

using Base.Iterators
using Distributions: Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG, rand!
using ProgressMeter: Progress, next!

# Automatic differentiation
using ADTypes: AbstractADType, AutoZygote, AutoEnzyme
import DifferentiationInterface as DI
const DEFAULT_AD_BACKEND = AutoZygote()

include("bibliography.jl")
include("input_augmentation.jl")
include("gradient.jl")
include("gradcam.jl")

export Gradient, InputTimesGradient
export NoiseAugmentation, SmoothGrad
export InterpolationAugmentation, IntegratedGradients
export GradCAM
export backend

end # module
