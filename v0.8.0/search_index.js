var documenterSearchIndex = {"docs":
[{"location":"api/#Basic-API","page":"API Reference","title":"Basic API","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"All methods in ExplainableAI.jl work by calling analyze on an input and an analyzer:","category":"page"},{"location":"api/","page":"API Reference","title":"API Reference","text":"analyze\nExplanation","category":"page"},{"location":"api/#XAIBase.analyze","page":"API Reference","title":"XAIBase.analyze","text":"analyze(input, method)\nanalyze(input, method, output_selection)\n\nApply the analyzer method for the given input, returning an Explanation. If output_selection is specified, the explanation will be calculated for that output. Otherwise, the output with the highest activation is automatically chosen.\n\nSee also Explanation.\n\nKeyword arguments\n\nadd_batch_dim: add batch dimension to the input without allocating. Default is false.\n\n\n\n\n\n","category":"function"},{"location":"api/#XAIBase.Explanation","page":"API Reference","title":"XAIBase.Explanation","text":"Explanation(val, output, output_selection, analyzer, heatmap, extras)\n\nReturn type of analyzers when calling analyze.\n\nFields\n\nval: numerical output of the analyzer, e.g. an attribution or gradient\noutput: model output for the given analyzer input\noutput_selection: index of the output used for the explanation\nanalyzer: symbol corresponding the used analyzer, e.g. :Gradient or :LRP\nheatmap: symbol indicating a preset heatmapping style,   e.g. :attribution, :sensitivity or :cam\nextras: optional named tuple that can be used by analyzers   to return additional information.\n\n\n\n\n\n","category":"type"},{"location":"api/","page":"API Reference","title":"API Reference","text":"For heatmapping functionality, take a look at either VisionHeatmaps.jl or TextHeatmaps.jl. Both provide heatmap methods for visualizing explanations,  either for images or text, respectively.","category":"page"},{"location":"api/#Analyzers","page":"API Reference","title":"Analyzers","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Gradient\nInputTimesGradient\nSmoothGrad\nIntegratedGradients\nGradCAM","category":"page"},{"location":"api/#ExplainableAI.Gradient","page":"API Reference","title":"ExplainableAI.Gradient","text":"Gradient(model)\n\nAnalyze model by calculating the gradient of a neuron activation with respect to the input.\n\n\n\n\n\n","category":"type"},{"location":"api/#ExplainableAI.InputTimesGradient","page":"API Reference","title":"ExplainableAI.InputTimesGradient","text":"InputTimesGradient(model)\n\nAnalyze model by calculating the gradient of a neuron activation with respect to the input. This gradient is then multiplied element-wise with the input.\n\n\n\n\n\n","category":"type"},{"location":"api/#ExplainableAI.SmoothGrad","page":"API Reference","title":"ExplainableAI.SmoothGrad","text":"SmoothGrad(analyzer, [n=50, std=0.1, rng=GLOBAL_RNG])\nSmoothGrad(analyzer, [n=50, distribution=Normal(0, σ²=0.01), rng=GLOBAL_RNG])\n\nAnalyze model by calculating a smoothed sensitivity map. This is done by averaging sensitivity maps of a Gradient analyzer over random samples in a neighborhood of the input, typically by adding Gaussian noise with mean 0.\n\nReferences\n\nSmilkov et al., SmoothGrad: removing noise by adding noise\n\n\n\n\n\n","category":"function"},{"location":"api/#ExplainableAI.IntegratedGradients","page":"API Reference","title":"ExplainableAI.IntegratedGradients","text":"IntegratedGradients(analyzer, [n=50])\nIntegratedGradients(analyzer, [n=50])\n\nAnalyze model by using the Integrated Gradients method.\n\nReferences\n\nSundararajan et al., Axiomatic Attribution for Deep Networks\n\n\n\n\n\n","category":"function"},{"location":"api/#ExplainableAI.GradCAM","page":"API Reference","title":"ExplainableAI.GradCAM","text":"GradCAM(feature_layers, adaptation_layers)\n\nCalculates the Gradient-weighted Class Activation Map (GradCAM). GradCAM provides a visual explanation of the regions with significant neuron importance for the model's classification decision.\n\nParameters\n\nfeature_layers: The layers of a convolutional neural network (CNN) responsible for extracting feature maps.\nadaptation_layers: The layers of the CNN used for adaptation and classification.\n\nNote\n\nFlux is not required for GradCAM.  GradCAM is compatible with a wide variety of CNN model-families.\n\nReferences\n\nSelvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\n\n\n\n\n\n","category":"type"},{"location":"api/#Input-augmentations","page":"API Reference","title":"Input augmentations","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"SmoothGrad and IntegratedGradients are special cases of the input augmentations  NoiseAugmentation and InterpolationAugmentation,  which can be applied as a wrapper to any analyzer:","category":"page"},{"location":"api/","page":"API Reference","title":"API Reference","text":"NoiseAugmentation\nInterpolationAugmentation","category":"page"},{"location":"api/#ExplainableAI.NoiseAugmentation","page":"API Reference","title":"ExplainableAI.NoiseAugmentation","text":"NoiseAugmentation(analyzer, n, [std=1, rng=GLOBAL_RNG])\nNoiseAugmentation(analyzer, n, distribution, [rng=GLOBAL_RNG])\n\nA wrapper around analyzers that augments the input with n samples of additive noise sampled from distribution. This input augmentation is then averaged to return an Explanation.\n\n\n\n\n\n","category":"type"},{"location":"api/#ExplainableAI.InterpolationAugmentation","page":"API Reference","title":"ExplainableAI.InterpolationAugmentation","text":"InterpolationAugmentation(model, [n=50])\n\nA wrapper around analyzers that augments the input with n steps of linear interpolation between the input and a reference input (typically zero(input)). The gradients w.r.t. this augmented input are then averaged and multiplied with the difference between the input and the reference input.\n\n\n\n\n\n","category":"type"},{"location":"api/#Index","page":"API Reference","title":"Index","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"EditURL = \"../literate/heatmapping.jl\"","category":"page"},{"location":"generated/heatmapping/#docs-heatmapping","page":"Heatmapping","title":"Heatmapping","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Since numerical explanations are not very informative at first sight, we can visualize them by computing a heatmap, using either VisionHeatmaps.jl or TextHeatmaps.jl.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"This page showcases different options and preset for heatmapping, building on the basics shown in the Getting started section.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"We start out by loading the same pre-trained LeNet5 model and MNIST input data:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"using ExplainableAI\nusing VisionHeatmaps\nusing Flux\n\nusing BSON # hide\nmodel = BSON.load(\"../model.bson\", @__MODULE__)[:model] # hide\nmodel","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"using MLDatasets\nusing ImageCore, ImageIO, ImageShow\n\nindex = 10\nx, y = MNIST(Float32, :test)[10]\ninput = reshape(x, 28, 28, 1, :)\n\nimg = convert2image(MNIST, x)","category":"page"},{"location":"generated/heatmapping/#Automatic-heatmap-presets","page":"Heatmapping","title":"Automatic heatmap presets","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"The function heatmap automatically applies common presets for each method.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Since InputTimesGradient computes attributions, heatmaps are shown in a blue-white-red color scheme. Gradient methods however are typically shown in grayscale:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"analyzer = Gradient(model)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"analyzer = InputTimesGradient(model)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/heatmapping/#Custom-heatmap-settings","page":"Heatmapping","title":"Custom heatmap settings","text":"","category":"section"},{"location":"generated/heatmapping/#Color-schemes","page":"Heatmapping","title":"Color schemes","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"We can partially or fully override presets by passing keyword arguments to heatmap. For example, we can use a custom color scheme from ColorSchemes.jl using the keyword argument colorscheme:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"using ColorSchemes\n\nexpl = analyze(input, analyzer)\nheatmap(expl; colorscheme=:jet)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; colorscheme=:inferno)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Refer to the ColorSchemes.jl catalogue for a gallery of available color schemes.","category":"page"},{"location":"generated/heatmapping/#docs-heatmap-reduce","page":"Heatmapping","title":"Color channel reduction","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Explanations have the same dimensionality as the inputs to the classifier. For images with multiple color channels, this means that the explanation also has a \"color channel\" dimension.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"The keyword argument reduce can be used to reduce this dimension to a single scalar value for each pixel. The following presets are available:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":":sum: sum up color channels (default setting)\n:norm: compute 2-norm over the color channels\n:maxabs: compute maximum(abs, x) over the color channels","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; reduce=:sum)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; reduce=:norm)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; reduce=:maxabs)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"In this example, the heatmaps look identical. Since MNIST only has a single color channel, there is no need for color channel reduction.","category":"page"},{"location":"generated/heatmapping/#docs-heatmap-rangescale","page":"Heatmapping","title":"Mapping explanations onto the color scheme","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"To map a color-channel-reduced explanation onto a color scheme, we first need to normalize all values to the range 0 1.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"For this purpose, two presets are available through the rangescale keyword argument:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":":extrema: normalize to the minimum and maximum value of the explanation\n:centered: normalize to the maximum absolute value of the explanation. Values of zero will be mapped to the center of the color scheme.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Depending on the color scheme, one of these presets may be more suitable than the other. The default color scheme for InputTimesGradient, seismic, is centered around zero, making :centered a good choice:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; rangescale=:centered)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; rangescale=:extrema)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"However, for the inferno color scheme, which is not centered around zero, :extrema leads to a heatmap with higher contrast.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; rangescale=:centered, colorscheme=:inferno)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap(expl; rangescale=:extrema, colorscheme=:inferno)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"For the full list of heatmap keyword arguments, refer to the heatmap documentation.","category":"page"},{"location":"generated/heatmapping/#overlay","page":"Heatmapping","title":"Heatmap overlays","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Heatmaps can be overlaid onto the input image using the heatmap_overlay function from VisionHeatmaps.jl. This can be useful for visualizing the relevance of specific regions of the input:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap_overlay(expl, img)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"The alpha value of the heatmap can be adjusted using the alpha keyword argument:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap_overlay(expl, img; alpha=0.3)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"All previously discussed keyword arguments for heatmap can also be used with heatmap_overlay:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmap_overlay(expl, img; alpha=0.7, colorscheme=:inferno, rangescale=:extrema)","category":"page"},{"location":"generated/heatmapping/#docs-heatmapping-batches","page":"Heatmapping","title":"Heatmapping batches","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Heatmapping also works with input batches. Let's demonstrate this by using a batch of 25 images from the MNIST dataset:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"xs, ys = MNIST(Float32, :test)[1:25]\nbatch = reshape(xs, 28, 28, 1, :); # reshape to WHCN format\nnothing #hide","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"The heatmap function automatically recognizes that the explanation is batched and returns a Vector of images:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"heatmaps = heatmap(batch, analyzer)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Image.jl's mosaic function can used to display them in a grid:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"mosaic(heatmaps; nrow=5)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"When heatmapping batches, the mapping to the color scheme is applied per sample. For example, rangescale=:extrema will normalize each heatmap to the minimum and maximum value of each sample in the batch. This ensures that heatmaps don't depend on other samples in the batch.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"If this bevahior is not desired, heatmap can be called with the keyword-argument process_batch=true:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"expl = analyze(batch, analyzer)\nheatmaps = heatmap(expl; process_batch=true)\nmosaic(heatmaps; nrow=5)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"This can be useful when comparing heatmaps for fixed output neurons:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"expl = analyze(batch, analyzer, 7) # explain digit \"6\"\nheatmaps = heatmap(expl; process_batch=true)\nmosaic(heatmaps; nrow=5)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"note: Output type consistency\nTo obtain a singleton Vector containing a single heatmap for non-batched inputs, use the heatmap keyword argument unpack_singleton=false.","category":"page"},{"location":"generated/heatmapping/#Processing-heatmaps","page":"Heatmapping","title":"Processing heatmaps","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Heatmapping makes use of the Julia-based image processing ecosystem Images.jl.","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"If you want to further process heatmaps, you may benefit from reading about some fundamental conventions that the ecosystem utilizes that are different from how images are typically represented in OpenCV, MATLAB, ImageJ or Python.","category":"page"},{"location":"generated/heatmapping/#Saving-heatmaps","page":"Heatmapping","title":"Saving heatmaps","text":"","category":"section"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"Since heatmaps are regular Images.jl images, they can be saved as such:","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"using FileIO\n\nimg = heatmap(input, analyzer)\nsave(\"heatmap.png\", img)","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"","category":"page"},{"location":"generated/heatmapping/","page":"Heatmapping","title":"Heatmapping","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"EditURL = \"../literate/augmentations.jl\"","category":"page"},{"location":"generated/augmentations/#docs-augmentations","page":"Input augmentations","title":"Analyzer augmentations","text":"","category":"section"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"All analyzers implemented in ExplainableAI.jl can be augmented by two types of augmentations: NoiseAugmentations and InterpolationAugmentations. These augmentations are wrappers around analyzers that modify the input before passing it to the analyzer.","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"We build on the basics shown in the Getting started section and start out by loading the same pre-trained LeNet5 model and MNIST input data:","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"using ExplainableAI\nusing VisionHeatmaps\nusing Flux\n\nusing BSON # hide\nmodel = BSON.load(\"../model.bson\", @__MODULE__)[:model] # hide\nmodel","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"using MLDatasets\nusing ImageCore, ImageIO, ImageShow\n\nindex = 10\nx, y = MNIST(Float32, :test)[10]\ninput = reshape(x, 28, 28, 1, :)\n\nconvert2image(MNIST, x)","category":"page"},{"location":"generated/augmentations/#Noise-augmentation","page":"Input augmentations","title":"Noise augmentation","text":"","category":"section"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"The NoiseAugmentation wrapper computes explanations averaged over noisy inputs. Let's demonstrate this on the Gradient analyzer. First, we compute the heatmap of an explanation without augmentation:","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"analyzer = Gradient(model)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"Now we wrap the analyzer in a NoiseAugmentation with 10 samples of noise. By default, the noise is sampled from a Gaussian distribution with mean 0 and standard deviation 1.","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"analyzer = NoiseAugmentation(Gradient(model), 50)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"Note that a higher sample size is desired, as it will lead to a smoother heatmap. However, this comes at the cost of a longer computation time.","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"We can also set the standard deviation of the Gaussian distribution:","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"analyzer = NoiseAugmentation(Gradient(model), 50, 0.1)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"When used with a Gradient analyzer, this is equivalent to SmoothGrad:","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"analyzer = SmoothGrad(model, 50)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"We can also use any distribution from Distributions.jl, for example Poisson noise with rate lambda=05:","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"using Distributions\n\nanalyzer = NoiseAugmentation(Gradient(model), 50, Poisson(0.5))\nheatmap(input, analyzer)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"Is is also possible to define your own distributions or mixture distributions.","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"NoiseAugmentation can be combined with any analyzer type from the Julia-XAI ecosystem, for example LRP from RelevancePropagation.jl.","category":"page"},{"location":"generated/augmentations/#Integration-augmentation","page":"Input augmentations","title":"Integration augmentation","text":"","category":"section"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"The InterpolationAugmentation wrapper computes explanations averaged over n steps of linear interpolation between the input and a reference input, which is set to zero(input) by default:","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"analyzer = InterpolationAugmentation(Gradient(model), 50)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"When used with a Gradient analyzer, this is equivalent to IntegratedGradients:","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"analyzer = IntegratedGradients(model, 50)\nheatmap(input, analyzer)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"To select a different reference input, pass it to the analyze function using the keyword argument input_ref. Note that this is an arbitrary example for the sake of demonstration.","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"matrix_of_ones = ones(Float32, size(input))\n\nanalyzer = InterpolationAugmentation(Gradient(model), 50)\nexpl = analyzer(input; input_ref=matrix_of_ones)\nheatmap(expl)","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"Once again, InterpolationAugmentation can be combined with any analyzer type from the Julia-XAI ecosystem, for example LRP from RelevancePropagation.jl.","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"","category":"page"},{"location":"generated/augmentations/","page":"Input augmentations","title":"Input augmentations","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"EditURL = \"../literate/example.jl\"","category":"page"},{"location":"generated/example/#docs-getting-started","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"note: Note\nThis package is part of a wider Julia XAI ecosystem. For an introduction to this ecosystem, please refer to the Getting started guide.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"For this first example, we already have loaded a pre-trained LeNet5 model to look at explanations on the MNIST dataset.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"using ExplainableAI\nusing Flux\n\nusing BSON # hide\nmodel = BSON.load(\"../model.bson\", @__MODULE__)[:model] # hide\nmodel","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"note: Supported models\nExplainableAI.jl can be used on any differentiable classifier.","category":"page"},{"location":"generated/example/#Preparing-the-input-data","page":"Getting started","title":"Preparing the input data","text":"","category":"section"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"We use MLDatasets to load a single image from the MNIST dataset:","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"using MLDatasets\nusing ImageCore, ImageIO, ImageShow\n\nindex = 10\nx, y = MNIST(Float32, :test)[10]\n\nconvert2image(MNIST, x)","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"By convention in Flux.jl, this input needs to be resized to WHCN format by adding a color channel and batch dimensions.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"input = reshape(x, 28, 28, 1, :);\nnothing #hide","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"note: Input format\nFor any explanation of a model, ExplainableAI.jl assumes the batch dimension to come last in the input.For the purpose of heatmapping, the input is assumed to be in WHCN order (width, height, channels, batch), which is Flux.jl's convention.","category":"page"},{"location":"generated/example/#Explanations","page":"Getting started","title":"Explanations","text":"","category":"section"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"We can now select an analyzer of our choice and call analyze to get an Explanation:","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"analyzer = InputTimesGradient(model)\nexpl = analyze(input, analyzer);\nnothing #hide","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"The return value expl is of type Explanation and bundles the following data:","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"expl.val: numerical output of the analyzer, e.g. an attribution or gradient\nexpl.output: model output for the given analyzer input\nexpl.output_selection: index of the output used for the explanation\nexpl.analyzer: symbol corresponding the used analyzer, e.g. :Gradient or :LRP\nexpl.heatmap: symbol indicating a preset heatmapping style,   e.g. :attibution, :sensitivity or :cam\nexpl.extras: optional named tuple that can be used by analyzers   to return additional information.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"We used InputTimesGradient, so expl.analyzer is :InputTimesGradient.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"expl.analyzer","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"By default, the explanation is computed for the maximally activated output neuron. Since our digit is a 9 and Julia's indexing is 1-based, the output neuron at index 10 of our trained model is maximally activated.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"Finally, we obtain the result of the analyzer in form of an array.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"expl.val","category":"page"},{"location":"generated/example/#Heatmapping-basics","page":"Getting started","title":"Heatmapping basics","text":"","category":"section"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"Since the array expl.val is not very informative at first sight, we can visualize Explanations by computing a heatmap using either VisionHeatmaps.jl or TextHeatmaps.jl.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"using VisionHeatmaps\n\nheatmap(expl)","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"If we are only interested in the heatmap, we can combine analysis and heatmapping into a single function call:","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"heatmap(input, analyzer)","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"For a more detailed explanation of the heatmap function, refer to the heatmapping section.","category":"page"},{"location":"generated/example/#docs-analyzers-list","page":"Getting started","title":"List of analyzers","text":"","category":"section"},{"location":"generated/example/#Neuron-selection","page":"Getting started","title":"Neuron selection","text":"","category":"section"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"By passing an additional index to our call to analyze, we can compute an explanation with respect to a specific output neuron. Let's see why the output wasn't interpreted as a 4 (output neuron at index 5)","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"expl = analyze(input, analyzer, 5)\nheatmap(expl)","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"This heatmap shows us that the \"upper loop\" of the hand-drawn 9 has negative relevance with respect to the output neuron corresponding to digit 4!","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"note: Note\nThe output neuron can also be specified when calling heatmap:heatmap(input, analyzer, 5)","category":"page"},{"location":"generated/example/#Analyzing-batches","page":"Getting started","title":"Analyzing batches","text":"","category":"section"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"ExplainableAI also supports explanations of input batches:","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"batchsize = 20\nxs, _ = MNIST(Float32, :test)[1:batchsize]\nbatch = reshape(xs, 28, 28, 1, :) # reshape to WHCN format\nexpl = analyze(batch, analyzer);\nnothing #hide","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"This will return a single Explanation expl for the entire batch. Calling heatmap on expl will detect the batch dimension and return a vector of heatmaps.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"heatmap(expl)","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"For more information on heatmapping batches, refer to the heatmapping documentation.","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"","category":"page"},{"location":"generated/example/","page":"Getting started","title":"Getting started","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ExplainableAI","category":"page"},{"location":"#ExplainableAI.jl","page":"Home","title":"ExplainableAI.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Explainable AI methods in Julia.","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Note\nThis package is part of a wider Julia XAI ecosystem. For an introduction to this ecosystem, please refer to the  Getting started guide.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install this package and its dependencies, open the Julia REPL and run ","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]add ExplainableAI","category":"page"},{"location":"#Manual","page":"Home","title":"Manual","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"generated/example.md\",\n    \"generated/heatmapping.md\",\n    \"generated/augmentations.md\",\n]\nDepth = 3","category":"page"},{"location":"#API-reference","page":"Home","title":"API reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"api.md\"]\nDepth = 2","category":"page"}]
}
