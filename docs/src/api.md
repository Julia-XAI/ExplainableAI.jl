# Basics
All methods in ExplainableAI.jl work by calling `analyze` on an input and an analyzer:
```@docs
analyze
Explanation
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

# Layer-wise Relevance Propagation
## LRP rules
```@docs
ZeroRule
EpsilonRule
GammaRule
WSquareRule
FlatRule
AlphaBetaRule
ZPlusRule
ZBoxRule
PassRule
GeneralizedGammaRule
```

## Custom rules 
These utilities can be used to define custom rules without writing boilerplate code.
To extend these functions, explicitly `import` them: 
```@docs
ExplainableAI.modify_input
ExplainableAI.modify_denominator
ExplainableAI.modify_parameters
ExplainableAI.modify_weight
ExplainableAI.modify_bias
ExplainableAI.modify_layer
ExplainableAI.is_compatible
```
Compatibility settings:
```@docs
LRP_CONFIG.supports_layer
LRP_CONFIG.supports_activation
```

## Composites
```@docs
Composite
```

### [Composite primitives](@id composite_primitive_api)
Composite primitives that apply a single rule:
```@docs
LayerMap
GlobalMap
RangeMap
FirstLayerMap
LastLayerMap
```

Composite primitives that apply a set of rules to multiple layers:
```@docs
GlobalTypeMap
RangeTypeMap
FirstLayerTypeMap
LastLayerTypeMap
FirstNTypeMap
```

### [Default composites](@id default_composite_api)
```@docs
EpsilonGammaBox
EpsilonPlus
EpsilonAlpha2Beta1
EpsilonPlusFlat
EpsilonAlpha2Beta1Flat
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
