# Basics
All methods in ExplainableAI.jl work by calling `analyze` on an input and an analyzer:
```@docs
analyze
heatmap
```

# Analyzers
```@docs
LRP
Gradient
InputTimesGradient
SmoothGrad
IntegratedGradients
```

`SmoothGrad` and `IntegratedGradients` are special cases of the input augmentation wrappers `NoiseAugmentation` and `InterpolationAugmentation`, which can be applied as a wrapper to any analyzer:
```@docs
NoiseAugmentation
InterpolationAugmentation
```

# LRP
## Rules
```@docs
ZeroRule
EpsilonRule
GammaRule
WSquareRule
AlphaBetaRule
FlatRule
ZBoxRule
PassRule
```

## Custom rules 
These utilities can be used to define custom rules without writing boilerplate code:
```@docs
modify_input
modify_denominator
modify_param!
modify_layer!
check_compat
LRP_CONFIG.supports_layer
LRP_CONFIG.supports_activation
```

# Utilities
```@docs
strip_softmax
flatten_model
canonize
```

# Index
```@index
```
