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
```

`SmoothGrad` is a special case of `NoiseAugmentation`, which can be applied as a wrapper to any analyzer:
```@docs
NoiseAugmentation
```

# LRP
## Rules
```@docs
ZeroRule
GammaRule
EpsilonRule
ZBoxRule
```

## Custom rules 
These utilities can be used to define custom rules without writing boilerplate code:
```@docs
modify_denominator
modify_params
modify_layer
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
