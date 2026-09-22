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
model = VGG(19, pretrain=true).layers

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

# Default pipelines with the 0.1% and 99.9% percentiles clipped after pooling
pipe_signed = SumPooling() |> PercentileClip() |> CenteredNormalization() |> Colormap(:berlin)
pipe_unsigned = NormPooling() |> PercentileClip() |> ExtremaNormalization() |> Colormap(:batlow)

for (name, method) in methods
    @info "Generating $name assets..."
    analyzer = method(model)

    # Max activated neuron corresponds to "castle"
    attr = analyze(input, analyzer)
    pipe = attr.pooling isa SignedPooling ? pipe_signed : pipe_unsigned
    save("castle_$name.png", only(heatmap(attr, pipe)))

    # Output neuron 920 corresponds to "street sign"
    attr = analyze(input, analyzer, 920)
    save("streetsign_$name.png", only(heatmap(attr, pipe)))
end
