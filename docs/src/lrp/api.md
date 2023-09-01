
# LRP analyzer
Refer to [`LRP`](@ref) for documentation on the LRP analyzer.

# [LRP rules](@id api-lrp-rules)
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

For [manual rule assignment](@ref docs-composites-manual), 
use `ChainTuple` and `ParallelTuple`, matching the model structure:
```@docs
ChainTuple
ParallelTuple
```

# Composites
## Applying composites
```@docs
Composite
lrp_rules
```

## [Composite primitives](@id api-composite-primitives)
### Mapping layers to rules
Composite primitives that apply a single rule:
```@docs
LayerMap
GlobalMap
RangeMap
FirstLayerMap
LastLayerMap
```

To apply `LayerMap` to nested Flux Chains or `Parallel` layers, 
make use of `show_layer_indices`:
```@docs
show_layer_indices
```

### Mapping layers to rules based on type
Composite primitives that apply rules based on the layer type:
```@docs
GlobalTypeMap
RangeTypeMap
FirstLayerTypeMap
LastLayerTypeMap
FirstNTypeMap
```

### Union types for composites
The following exported union types types can be used to define TypeMaps:
```@docs
ConvLayer
PoolingLayer
DropoutLayer
ReshapingLayer
NormalizationLayer
```

## [Composite presets](@id api-composite-presets)
```@docs
EpsilonGammaBox
EpsilonPlus
EpsilonAlpha2Beta1
EpsilonPlusFlat
EpsilonAlpha2Beta1Flat
```

# Custom rules 
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

# Index
```@index
```
