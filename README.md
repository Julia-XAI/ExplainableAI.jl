![ExplainableAI.jl][banner-img]
___

*Formerly known as ExplainabilityMethods.jl*

| **Documentation**                                                     | **Build Status**                                      | **DOI**                 |
|:--------------------------------------------------------------------- |:----------------------------------------------------- |:----------------------- |
| [![][docs-stab-img]][docs-stab-url] [![][docs-dev-img]][docs-dev-url] | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url] | [![][doi-img]][doi-url] |

Explainable AI in Julia using [Flux.jl](https://fluxml.ai).

This package implements interpretability methods and visualizations for neural networks, similar to [Captum][captum-repo] and [Zennit][zennit-repo] for PyTorch and [iNNvestigate][innvestigate-repo] for Keras models. 

## Installation 
This package supports Julia ≥1.6. To install it, open the Julia REPL and run 
```julia-repl
julia> ]add ExplainableAI
```

## Example
Let's use LRP to explain why an image of a castle gets classified as such using a pre-trained VGG16 model from [Metalhead.jl](https://github.com/FluxML/Metalhead.jl):
![][castle]
```julia
using ExplainableAI
using Flux
using Metalhead
using FileIO

# Load model
model = VGG(16, pretrain=true).layers
model = strip_softmax(flatten_chain(model))

# Load input
img = load("castle.jpg")
input = preprocess_imagenet(img)
input = reshape(input, 224, 224, 3, :)  # reshape to WHCN format

# Run XAI method
analyzer = LRP(model)
expl = analyze(input, analyzer)         # or: expl = analyzer(input)

# Show heatmap
heatmap(expl)

# Or analyze & show heatmap directly
heatmap(input, analyzer)
```

We can also get an explanation for the activation of the output neuron corresponding to the "street sign" class by specifying the corresponding output neuron position `920`:
```julia
analyze(input, analyzer, 920)  # for explanation 
heatmap(input, analyzer, 920)  # for heatmap
```
Heatmaps for all implemented analyzers are shown in the following table. Red color indicate regions of positive relevance towards the selected class, whereas regions in blue are of negative relevance.

| **Analyzer**          | **Heatmap for class "castle"** |**Heatmap for class "street sign"** |
|:--------------------- |:------------------------------ |:---------------------------------- |
| `LRP` composite       | ![][castle-lrp-comp]           | ![][streetsign-lrp-comp]           |
| `LRP`                 | ![][castle-lrp]                | ![][streetsign-lrp]                |
| `InputTimesGradient`  | ![][castle-ixg]                | ![][streetsign-ixg]                |
| `Gradient`            | ![][castle-grad]               | ![][streetsign-grad]               |
| `SmoothGrad`          | ![][castle-smoothgrad]         | ![][streetsign-smoothgrad]         |
| `IntegratedGradients` | ![][castle-intgrad]            | ![][streetsign-intgrad]            |

The code used to generate these heatmaps can be found [here][asset-code].

## Methods
Currently, the following analyzers are implemented:

```
├── Gradient
├── InputTimesGradient
├── SmoothGrad
├── IntegratedGradients
└── LRP
    ├── ZeroRule
    ├── EpsilonRule
    ├── GammaRule
    ├── WSquareRule
    ├── FlatRule
    ├── ZBoxRule
    ├── AlphaBetaRule
    └── PassRule
```

One of the design goals of ExplainableAI.jl is extensibility.
Individual LRP rules [can be composed][docs-composites] and are easily extended by [custom rules][docs-custom-rules].

## Video demonstration
Check out our [JuliaCon 2022 talk][juliacon-url] for a demonstration of the package.
[![][juliacon-img]][juliacon-url]

## Roadmap
In the future, we would like to include:
- [PatternNet](https://arxiv.org/abs/1705.05598)
- [DeepLift](https://arxiv.org/abs/1704.02685)
- [LIME](https://arxiv.org/abs/1602.04938)
- Shapley values via  [ShapML.jl](https://github.com/nredell/ShapML.jl)

Contributions are welcome!

## Acknowledgements
> Adrian Hill acknowledges support by the Federal Ministry of Education and Research (BMBF) for the Berlin Institute for the Foundations of Learning and Data (BIFOLD) (01IS18037A).

[banner-img]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/banner.png

[asset-code]: https://github.com/adrhill/ExplainableAI.jl/blob/gh-pages/assets/heatmaps/readme_assets.jl
[castle]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg
[castle-lrp]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRP.png
[castle-lrp-comp]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRP_composite.png
[castle-ixg]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_InputTimesGradient.png
[castle-grad]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_Gradient.png
[castle-smoothgrad]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_SmoothGrad.png
[castle-intgrad]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_IntegratedGradients.png
[streetsign-lrp]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRP.png
[streetsign-lrp-comp]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRP_composite.png
[streetsign-ixg]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_InputTimesGradient.png
[streetsign-grad]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_Gradient.png
[streetsign-smoothgrad]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_SmoothGrad.png
[streetsign-intgrad]: https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_IntegratedGradients.png

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

[juliacon-img]: http://img.youtube.com/vi/p5dg3vdmlvI/0.jpg
[juliacon-url]: https://www.youtube.com/watch?v=p5dg3vdmlvI

[captum-repo]: https://github.com/pytorch/captum
[zennit-repo]: https://github.com/chr5tphr/zennit
[innvestigate-repo]: https://github.com/albermax/innvestigate
