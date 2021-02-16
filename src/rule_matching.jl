CommonLayer = Union{Dense,Conv,MaxPool}
"""
Match rules based on position in the `Flux.Chain`.
"""
LRP(l::CommonLayer, a, R, ::Val(:FirstPixels)) = LRP_zB(l, a, R)
LRP(l::CommonLayer, a, R, ::Val(:Middle)) = LRP_ϵ(l, a, R)
LRP(l::CommonLayer, a, R, ::Val(:Lower)) = LRP_γ(l, a, R)
LRP(l::CommonLayer, a, R, ::Val(:Upper)) = LRP_0(l, a, R)
