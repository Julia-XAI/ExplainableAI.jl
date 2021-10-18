![ExplainabilityMethods.jl][banner-img]
___

| **Documentation**                                                     | **Build Status**                                      |
|:--------------------------------------------------------------------- |:----------------------------------------------------- |
| [![][docs-stab-img]][docs-stab-url] [![][docs-dev-img]][docs-dev-url] | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url] |

Explainable AI in Julia using Flux.

## Installation 
To install this package and its dependencies, open the Julia REPL and run 
```julia-repl
julia> ]add https://github.com/adrhill/ExplainabilityMethods.jl
```

⚠️ This package is in early development, so expect frequent breaking changes. ⚠️

## Example
```julia
using Flux
using Metalhead
using ExplainabilityMethods

# Load model
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

[banner-img]: https://raw.githubusercontent.com/adrhill/ExplainabilityMethods.jl/gh-pages/assets/banner.png

[docs-stab-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stab-url]: https://adrhill.github.io/ExplainabilityMethods.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-main-blue.svg
[docs-dev-url]: https://adrhill.github.io/ExplainabilityMethods.jl/dev

[ci-img]: https://github.com/adrhill/ExplainabilityMethods.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/adrhill/ExplainabilityMethods.jl/actions

[codecov-img]: https://codecov.io/gh/adrhill/ExplainabilityMethods.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/adrhill/ExplainabilityMethods.jl
