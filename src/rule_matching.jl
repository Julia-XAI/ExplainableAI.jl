CommonLayer = Union{Dense,Conv,MaxPool}
"""
Match rules based on position in the `Flux.Chain` according to

> **Upper layers** have only approximately 4 000 neurons (i.e. on average 4 neurons per class), making it likely that the many concepts forming the different classes are entangled. Here, a propagation rule close to the function and its gradient (e.g. LRP-0) will be insensitive to these entanglements.
> **Middle layers** have a more disentangled representation, however, the stacking of many layers and the weight sharing in convolutions introduces spurious variations. LRP-ε filters out these spurious variations and retains only the most salient explanation factors.
> **Lower layers** are similar to middle layers, however, LRP-γ is more suitable here, as this rule tends to spread relevance uniformly to the whole feature rather than capturing the contribution of every individual pixel. This makes the explanation more understandable for a human.

And in (10.3.2) on Handling Special Layers:
> **Input Layers** are different from intermediate layers as they do not receive ReLU activations as input but pixels or real values [...]. In this chapter, we made use of the zB-rule, which is suitable for pixels.
"""
LRP(l::CommonLayer, a, R, ::Val(:Upper)) = LRP_0(l, a, R)
LRP(l::CommonLayer, a, R, ::Val(:Middle)) = LRP_ϵ(l, a, R)
LRP(l::CommonLayer, a, R, ::Val(:Lower)) = LRP_γ(l, a, R)
LRP(l::CommonLayer, a, R, ::Val(:FirstPixels)) = LRP_zB(l, a, R)
