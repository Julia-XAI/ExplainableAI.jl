# ExplainabilityMethods.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adrhill.github.io/ExplainabilityMethods.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://adrhill.github.io/ExplainabilityMethods.jl/dev)
[![Build Status](https://github.com/adrhill/ExplainabilityMethods.jl/workflows/CI/badge.svg)](https://github.com/adrhill/ExplainabilityMethods.jl/actions)
[![Coverage](https://codecov.io/gh/adrhill/ExplainabilityMethods.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/adrhill/ExplainabilityMethods.jl)

Explainable AI (XAI) in Julia using Flux.

## Installation 
To install this package and its dependencies, open the Julia REPL and run 
```julia-repl
julia> ]add https://github.com/adrhill/ExplainabilityMethods.jl
```

⚠️ This package is in early development, so expect frequent breaking changes ⚠️

## Example
```julia
using Flux
using Metalhead
using ExplainabilityMethods

# Load VGG model
vgg = VGG19()
model = strip_softmax(vgg.layers)

# Run XAI method
analyzer = LRPEpsilon(model)
expl, out = analyze(img, analyzer) 

# Show heatmap
heatmap(expl)
```

Currently, the following analyzers are implemented:

```
├── Gradient
├── InputTimesGradient
└── LRP
    ├── LRPZero
    ├── LRPEpsilon
    └── LRPGamma
```

One of the design goals of ExplainabilityMethods.jl is extensibility.
Individual LRP rules like `ZeroRule`, `EpsilonRule`, `GammaRule` and `ZBoxRule` can be composed and are easily extended by custom rules.
