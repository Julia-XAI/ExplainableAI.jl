# Basic API
All methods in ExplainableAI.jl work by calling `analyze` on an input and an analyzer:
```@docs
analyze
Explanation
```

For heatmapping functionality, take a look at either
[VisionHeatmaps.jl](https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/) or
[TextHeatmaps.jl](https://julia-xai.github.io/XAIDocs/TextHeatmaps/stable/).
Both provide `heatmap` methods for visualizing explanations, 
either for images or text, respectively.

# Analyzers
```@docs
Gradient
InputTimesGradient
SmoothGrad
IntegratedGradients
GradCAM
```

# Input augmentations
`SmoothGrad` and `IntegratedGradients` are special cases of the input augmentations 
`NoiseAugmentation` and `InterpolationAugmentation`, 
which can be applied as a wrapper to any analyzer:
```@docs
NoiseAugmentation
InterpolationAugmentation
```

# Index
```@index
```

