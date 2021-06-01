module ExplainabilityMethods

using LinearAlgebra
using Flux
using Statistics: std

include("api/explain.jl")
include("methods/gradient.jl")
include("methods/lrp/rules.jl")

export explain, classify_and_explain

# rules
export Gradient
export LRP_0, LRP_γ, LRP_ϵ, LRP_zᴮ

end # module
