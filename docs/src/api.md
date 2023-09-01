# Basic API
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

# Input augmentations
`SmoothGrad` and `IntegratedGradients` are special cases of the input augmentation wrappers `NoiseAugmentation` and `InterpolationAugmentation`, which can be applied as a wrapper to any analyzer:
```@docs
NoiseAugmentation
InterpolationAugmentation
```

# Model preparation
```@docs
strip_softmax
canonize
flatten_model
```

# Input preprocessing
```@docs
preprocess_imagenet
```

# Index
```@index
```
