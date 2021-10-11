"""
    LRPLayer(layer, rule)

Struct assigning a LRP-rule to a Flux layer.
"""
struct LRPLayer{L, R} where {R<:AbstractLRPRule}
    layer::L
    rule::R
end
# For convenience, calls to LRPLayer are forwarded to its LRP-rule.
(l::LRPLayer)(aₖ, Rₖ₊₁) = l.rule(l.layer, aₖ, Rₖ₊₁)

# This package automatically matches rules based on position in the `Flux.Chain`
# according to "Layer-Wise Relevance Propagation: An Overview" chapter (10.2.2):
# * **Upper layers** have only approximately 4 000 neurons (i.e. on average 4 neurons per class),
#     making it likely that the many concepts forming the different classes are entangled.
#     Here, a propagation rule close to the function and its gradient (e.g. LRP-0)
#     will be insensitive to these entanglements.
# * **Middle layers** have a more disentangled representation, however,
#     the stacking of many layers and the weight sharing in convolutions introduces spurious variations.
#     LRP-ε filters out these spurious variations and retains only the most salient explanation factors.
# * **Lower layers** are similar to middle layers, however, LRP-γ is more suitable here,
#     as this rule tends to spread relevance uniformly to the whole feature
#     rather than capturing the contribution of every individual pixel.
#     This makes the explanation more understandable for a human.
#
# And in (10.3.2) on Handling Special Layers:
#
# * **Input Layers** are different from intermediate layers as they do not receive
#     ReLU activations as input but pixels or real values [...].
#     In this chapter, we made use of the zB-rule, which is suitable for pixels.
#
# We implement this as custom constructors for `LRPLayer`:

LRPLayer(l, ::Val(:Upper)) = LRPLayer(l, ZeroRule())
LRPLayer(l, ::Val(:Middle)) = LRPLayer(l, EpsilonRule())
LRPLayer(l, ::Val(:Lower)) = LRPLayer(l, GammaRule())
LRPLayer(l, ::Val(:FirstPixels)) = LRPLayer(l, ZBoxRule())
