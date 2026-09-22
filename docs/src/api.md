# Basic API
All methods in ExplainableAI.jl work by calling `analyze` on an input and an analyzer:
```@docs
analyze
Attribution
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

All gradient-based analyzers use AD backends from
[ADTypes.jl](https://github.com/SciML/ADTypes.jl) via
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl),
which can be selected on construction and queried via `backend`:
```@docs
backend
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
