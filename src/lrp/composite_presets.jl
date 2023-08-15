"""
    EpsilonGammaBox(low, high; [epsilon=1.0f-6, gamma=0.25f0])

Composite using the following primitives:
```julia-repl
julia> EpsilonGammaBox(-3.0f0, 3.0f0)
$(repr("text/plain", EpsilonGammaBox(-3.0f0, 3.0f0)))
```
"""
function EpsilonGammaBox(low, high; epsilon=1.0f-6, gamma=0.25f0)
    return Composite(
        GlobalTypeRule(
            ConvLayer        => GammaRule(gamma),
            Dense            => EpsilonRule(epsilon),
            DropoutLayer     => PassRule(),
            ReshapingLayer   => PassRule(),
            typeof(identity) => PassRule(),
        ),
        FirstLayerTypeRule(ConvLayer => ZBoxRule(low, high)),
    )
end

"""
    EpsilonPlus(; [epsilon=1.0f-6])

Composite using the following primitives:
```julia-repl
julia> EpsilonPlus()
$(repr("text/plain", EpsilonPlus()))
```
"""
function EpsilonPlus(; epsilon=1.0f-6)
    return Composite(
        GlobalTypeRule(
            ConvLayer        => ZPlusRule(),
            Dense            => EpsilonRule(epsilon),
            DropoutLayer     => PassRule(),
            ReshapingLayer   => PassRule(),
            typeof(identity) => PassRule(),
        ),
    )
end

"""
    EpsilonAlpha2Beta1(; [epsilon=1.0f-6])

Composite using the following primitives:
```julia-repl
julia> EpsilonAlpha2Beta1()
$(repr("text/plain", EpsilonAlpha2Beta1()))
```
"""
function EpsilonAlpha2Beta1(; epsilon=1.0f-6)
    return Composite(
        GlobalTypeRule(
            ConvLayer        => AlphaBetaRule(2.0f0, 1.0f0),
            Dense            => EpsilonRule(epsilon),
            DropoutLayer     => PassRule(),
            ReshapingLayer   => PassRule(),
            typeof(identity) => PassRule(),
        ),
    )
end

"""
    EpsilonPlusFlat(; [epsilon=1.0f-6])

Composite using the following primitives:
```julia-repl
julia> EpsilonPlusFlat()
$(repr("text/plain", EpsilonPlusFlat()))
```
"""
function EpsilonPlusFlat(; epsilon=1.0f-6)
    return Composite(
        GlobalTypeRule(
            ConvLayer        => ZPlusRule(),
            Dense            => EpsilonRule(epsilon),
            DropoutLayer     => PassRule(),
            ReshapingLayer   => PassRule(),
            typeof(identity) => PassRule(),
        ),
        FirstLayerTypeRule(ConvLayer => FlatRule(), Dense => FlatRule()),
    )
end

"""
    EpsilonAlpha2Beta1Flat(; [epsilon=1.0f-6])

Composite using the following primitives:
```julia-repl
julia> EpsilonAlpha2Beta1Flat()
$(repr("text/plain", EpsilonAlpha2Beta1Flat()))
```
"""
function EpsilonAlpha2Beta1Flat(; epsilon=1.0f-6)
    return Composite(
        GlobalTypeRule(
            ConvLayer        => AlphaBetaRule(2.0f0, 1.0f0),
            Dense            => EpsilonRule(epsilon),
            DropoutLayer     => PassRule(),
            ReshapingLayer   => PassRule(),
            typeof(identity) => PassRule(),
        ),
        FirstLayerTypeRule(ConvLayer => FlatRule(), Dense => FlatRule()),
    )
end
