module LayerwiseRelevancePropagation

using LinearAlgebra
using Flux
using Statistics: std

include("rules.jl")

# rules
export LRP_0, LRP_γ, LRP_ϵ, LRP_zᴮ

end # module
