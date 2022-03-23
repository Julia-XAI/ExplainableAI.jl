![ExplainableAI.jl][banner-img]
___

| **Documentation**                                                     | **Build Status**                                      | **DOI**                 |
|:--------------------------------------------------------------------- |:----------------------------------------------------- |:----------------------- |
| [![][docs-stab-img]][docs-stab-url] [![][docs-dev-img]][docs-dev-url] | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url] | [![][doi-img]][doi-url] |

Explainable AI in Julia using [Flux.jl](https://fluxml.ai).

## Installation 
To install this package and its dependencies, open the Julia REPL and run 
```julia-repl
julia> ]add ExplainableAI
```

⚠️ This package is still in early development, expect breaking changes. ⚠️

## Example
Let's use LRP to explain why an image of a cat gets classified as a cat:
```julia
using ExplainableAI
using Flux
using Metalhead

# Load model
vgg = VGG19()
model = strip_softmax(vgg.layers)

# Run XAI method
analyzer = LRPEpsilon(model)
expl = analyze(img, analyzer)

# Show heatmap
heatmap(expl)
```
![][heatmap]


## Methods
Currently, the following analyzers are implemented:

```
├── Gradient
├── InputTimesGradient
└── LRP
    ├── LRPZero
    ├── LRPEpsilon
    └── LRPGamma
```

One of the design goals of ExplainableAI.jl is extensibility.
Individual LRP rules like `ZeroRule`, `EpsilonRule`, `GammaRule` and `ZBoxRule` [can be composed][docs-composites] and are easily extended by [custom rules][docs-custom-rules].

## Acknowledgements
> Adrian Hill acknowledges support by the Federal Ministry of Education and Research (BMBF) for the Berlin Institute for the Foundations of Learning and Data (BIFOLD) (01IS18037A).

[banner-img]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/banner.png
[heatmap]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmap.png

[docs-stab-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stab-url]: https://adrhill.github.io/ExplainableAI.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-main-blue.svg
[docs-dev-url]: https://adrhill.github.io/ExplainableAI.jl/dev

[ci-img]: https://github.com/adrhill/ExplainableAI.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/adrhill/ExplainableAI.jl/actions

[codecov-img]: https://codecov.io/gh/adrhill/ExplainableAI.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/adrhill/ExplainableAI.jl

[docs-composites]: https://adrhill.github.io/ExplainableAI.jl/dev/generated/advanced_lrp/#Custom-LRP-composites
[docs-custom-rules]: https://adrhill.github.io/ExplainableAI.jl/dev/generated/advanced_lrp/#Custom-LRP-rules

[doi-img]: https://zenodo.org/badge/337430397.svg
[doi-url]: https://zenodo.org/badge/latestdoi/337430397
