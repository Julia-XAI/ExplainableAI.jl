![ExplainableAI.jl][banner-img]
___

|               |                                                                                                           |
|:--------------|:----------------------------------------------------------------------------------------------------------|
| Documentation | [![][docs-stab-img]][docs-stab-url] [![][docs-dev-img]][docs-dev-url] [![][changelog-img]][changelog-url] |
| Build Status  | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url]                                                     |
| Testing       | [![Aqua][aqua-img]][aqua-url] [![JET][jet-img]][jet-url]                                                  |
| Code Style    | [![Code Style: Blue][blue-img]][blue-url] [![ColPrac][colprac-img]][colprac-url]                          |
| Citation      | [![][doi-img]][doi-url]                                                                                   |

Explainable AI in Julia.

This package implements interpretability methods for black-box classifiers,
with an emphasis on local explanations and attribution maps in input space.
The only requirement for the model is that it is differentiable[^1].
It is similar to [Captum][captum-repo] and [Zennit][zennit-repo] for PyTorch 
and [iNNvestigate][innvestigate-repo] for Keras models.

[^1]: More specifically, models currently have to be differentiable with [Zygote.jl](https://github.com/FluxML/Zygote.jl).

## Installation 
This package supports Julia ≥1.10. To install it, open the Julia REPL and run 
```julia-repl
julia> ]add ExplainableAI
```

## Example
Let's explain why an image of a castle is classified as such by a vision model:

![][castle]

```julia
using ExplainableAI
using VisionHeatmaps         # visualization of explanations as heatmaps
using Zygote                 # load autodiff backend for gradient-based methods
using Flux, Metalhead        # pre-trained vision models in Flux
using DataAugmentation       # input preprocessing
using HTTP, FileIO, ImageIO  # load image from URL
using ImageInTerminal        # show heatmap in terminal

# Load & prepare model
model = VGG(16, pretrain=true)

# Load input
url = HTTP.URI("https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = load(url) 

# Preprocess input
mean = (0.485f0, 0.456f0, 0.406f0)
std  = (0.229f0, 0.224f0, 0.225f0)
tfm = CenterResizeCrop((224, 224)) |> ImageToTensor() |> Normalize(mean, std)
input = apply(tfm, Image(img))               # apply DataAugmentation transform
input = reshape(input.data, 224, 224, 3, :)  # unpack data and add batch dimension

# Run XAI method
analyzer = SmoothGrad(model)
expl = analyze(input, analyzer)  # or: expl = analyzer(input)
heatmap(expl)                    # show heatmap using VisionHeatmaps.jl
```

By default, explanations are computed for the class with the highest activation.
We can also compute explanations for a specific class, e.g. the one at output index 5:

```julia
analyze(input, analyzer, 5)  # for explanation 
heatmap(input, analyzer, 5)  # for heatmap
```

| **Analyzer**                                  | **Heatmap for class "castle"** |**Heatmap for class "street sign"** |
|:--------------------------------------------- |:------------------------------:|:----------------------------------:|
| `InputTimesGradient`                          | ![][castle-ixg]                | ![][streetsign-ixg]                |
| `Gradient`                                    | ![][castle-grad]               | ![][streetsign-grad]               |
| `SmoothGrad`                                  | ![][castle-smoothgrad]         | ![][streetsign-smoothgrad]         |
| `IntegratedGradients`                         | ![][castle-intgrad]            | ![][streetsign-intgrad]            |

> [!TIP]
> The heatmaps shown above were created using a VGG-16 vision model 
> from [Metalhead.jl](https://github.com/FluxML/Metalhead.jl)
> that was pre-trained on the [ImageNet](http://www.image-net.org/) dataset.
>
> Since ExplainableAI.jl can be used outside of Deep Learning models and [Flux.jl](https://github.com/FluxML/Flux.jl),
> we have omitted specific models and inputs from the code snippet above. 
> The full code used to generate the heatmaps can be found [here][asset-code].

Depending on the method, the applied heatmapping defaults differ:
sensitivity-based methods (e.g. `Gradient`) default to a grayscale color scheme,
whereas attribution-based methods (e.g. `InputTimesGradient`) default to a red-white-blue color scheme.
Red color indicates regions of positive relevance towards the selected class, 
whereas regions in blue are of negative relevance.
More information on heatmapping presets can be found in the [Julia-XAI documentation](https://julia-xai.github.io/XAIDocs/XAIDocs/dev/generated/heatmapping/).

> [!WARNING]
> ExplainableAI.jl used to contain Layer-wise Relevance Propagation (LRP).
> Since version `v0.7.0`, LRP is now available as part of a separate package in the Julia-XAI ecosystem,
> called [RelevancePropagation.jl](https://github.com/Julia-XAI/RelevancePropagation.jl).
>
> | **Analyzer**                                  | **Heatmap for class "castle"** |**Heatmap for class "street sign"** |
> |:--------------------------------------------- |:------------------------------:|:----------------------------------:|
> | `LRP` with `EpsilonPlus` composite            | ![][castle-comp-ep]            | ![][streetsign-comp-ep]            |
> | `LRP` with `EpsilonPlusFlat` composite        | ![][castle-comp-epf]           | ![][streetsign-comp-epf]           |
> | `LRP` with `EpsilonAlpha2Beta1` composite     | ![][castle-comp-eab]           | ![][streetsign-comp-eab]           |
> | `LRP` with `EpsilonAlpha2Beta1Flat` composite | ![][castle-comp-eabf]          | ![][streetsign-comp-eabf]          |
> | `LRP` with `EpsilonGammaBox` composite        | ![][castle-comp-egb]           | ![][streetsign-comp-egb]           |
> | `LRP` with `ZeroRule` (discouraged)           | ![][castle-lrp]                | ![][streetsign-lrp]                |

## Video Demonstration
Check out our talk at JuliaCon 2022 for a demonstration of the package.

[![][juliacon-img]][juliacon-url]

## Methods
Currently, the following analyzers are implemented:

* `Gradient`
* `InputTimesGradient`
* `SmoothGrad`
* `IntegratedGradients`
* `GradCAM`

One of the design goals of the [Julia-XAI ecosystem][juliaxai-docs] is extensibility.
To implement an XAI method, take a look at the [common interface
defined in XAIBase.jl][xaibase-docs].

## Roadmap
In the future, we would like to include:
- [PatternNet](https://arxiv.org/abs/1705.05598)
- [DeepLift](https://arxiv.org/abs/1704.02685)
- [LIME](https://arxiv.org/abs/1602.04938)
- Shapley values via  [ShapML.jl](https://github.com/nredell/ShapML.jl)

Contributions are welcome!

## Acknowledgements
> Adrian Hill acknowledges support by the Federal Ministry of Education and Research (BMBF) 
> for the Berlin Institute for the Foundations of Learning and Data (BIFOLD) (01IS18037A).

[banner-img]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/banner.png
[juliaxai-docs]: https://julia-xai.github.io/XAIDocs/
[xaibase-docs]: https://julia-xai.github.io/XAIDocs/XAIBase/


[asset-code]: https://github.com/Julia-XAI/ExplainableAI.jl/blob/gh-pages/assets/heatmaps/generate_assets.jl
[castle]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg

[castle-lrp]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRP.png
[castle-ixg]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_InputTimesGradient.png
[castle-grad]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_Gradient.png
[castle-smoothgrad]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_SmoothGrad.png
[castle-intgrad]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_IntegratedGradients.png
[castle-comp-egb]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonGammaBox.png
[castle-comp-ep]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonPlus.png
[castle-comp-epf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonPlusFlat.png
[castle-comp-eab]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonAlpha2Beta1.png
[castle-comp-eabf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonAlpha2Beta1Flat.png

[streetsign-lrp]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRP.png
[streetsign-ixg]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_InputTimesGradient.png
[streetsign-grad]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_Gradient.png
[streetsign-smoothgrad]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_SmoothGrad.png
[streetsign-intgrad]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_IntegratedGradients.png
[streetsign-comp-egb]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonGammaBox.png
[streetsign-comp-ep]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonPlus.png
[streetsign-comp-epf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonPlusFlat.png
[streetsign-comp-eab]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonAlpha2Beta1.png
[streetsign-comp-eabf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonAlpha2Beta1Flat.png

[docs-stab-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stab-url]: https://julia-xai.github.io/XAIDocs/ExplainableAI/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://julia-xai.github.io/ExplainableAI.jl/dev
[changelog-img]: https://img.shields.io/badge/news-changelog-yellow.svg
[changelog-url]: https://github.com/Julia-XAI/ExplainableAI.jl/blob/master/CHANGELOG.md

[ci-img]: https://github.com/Julia-XAI/ExplainableAI.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Julia-XAI/ExplainableAI.jl/actions
[codecov-img]: https://codecov.io/gh/Julia-XAI/ExplainableAI.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Julia-XAI/ExplainableAI.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl
[jet-img]: https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a
[jet-url]: https://github.com/aviatesk/JET.jl


[blue-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[blue-url]: https://github.com/invenia/BlueStyle
[colprac-img]: https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet
[colprac-url]: https://github.com/SciML/ColPrac

[docs-composites]: https://julia-xai.github.io/ExplainableAI.jl/stable/generated/lrp/composites/
[docs-custom-rules]: https://julia-xai.github.io/ExplainableAI.jl/stable/generated/lrp/custom_rules/

[doi-img]: https://zenodo.org/badge/337430397.svg
[doi-url]: https://zenodo.org/badge/latestdoi/337430397

[juliacon-img]: http://img.youtube.com/vi/p5dg3vdmlvI/0.jpg
[juliacon-url]: https://www.youtube.com/watch?v=p5dg3vdmlvI

[captum-repo]: https://github.com/pytorch/captum
[zennit-repo]: https://github.com/chr5tphr/zennit
[innvestigate-repo]: https://github.com/albermax/innvestigate
