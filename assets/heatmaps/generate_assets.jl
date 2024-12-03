using Pkg
Pkg.activate(@__DIR__)

using ExplainableAI
using RelevancePropagation
using VisionHeatmaps
using Zygote                 # load autodiff backend for gradient-based methods
using Flux, Metalhead        # pre-trained vision models in Flux
using DataAugmentation       # input preprocessing
using HTTP, FileIO, ImageIO  # load image from URL
using ImageInTerminal        # show heatmap in terminal

# Load & prepare model
model = VGG(16, pretrain=true).layers

# Load input
url = HTTP.URI("https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = load(url) 

# Preprocess input
mean = (0.485f0, 0.456f0, 0.406f0)
std  = (0.229f0, 0.224f0, 0.225f0)
tfm = CenterResizeCrop((224, 224)) |> ImageToTensor() |> Normalize(mean, std)
input = apply(tfm, Image(img))               # apply DataAugmentation transform
input = reshape(input.data, 224, 224, 3, :)  # unpack data and add batch dimension

# Run XAI methods
methods = Dict(
    "InputTimesGradient"        => InputTimesGradient,
    "Gradient"                  => Gradient,
    "SmoothGrad"                => SmoothGrad,
    "IntegratedGradients"       => IntegratedGradients,
    "LRP"                       => LRP,
    "LRPEpsilonGammaBox"        => model -> LRP(model, EpsilonGammaBox(-3.0f0, 3.0f0)),
    "LRPEpsilonPlus"            => model -> LRP(model, EpsilonPlus()),
    "LRPEpsilonAlpha2Beta1"     => model -> LRP(model, EpsilonAlpha2Beta1()),
    "LRPEpsilonPlusFlat"        => model -> LRP(model, EpsilonPlusFlat()),
    "LRPEpsilonAlpha2Beta1Flat" => model -> LRP(model, EpsilonAlpha2Beta1Flat()),
)

for (name, method) in methods
    @info "Generating $name assets..."
    analyzer = method(model)

    # Max activated neuron corresponds to "castle"
    h = heatmap(input, analyzer)
    save("castle_$name.png", h)

    # Output neuron 920 corresponds to "street sign"
    h = heatmap(input, analyzer, 920)
    save("streetsign_$name.png", h)
end
