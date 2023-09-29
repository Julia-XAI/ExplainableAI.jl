using ExplainableAI
using Metalhead                   # pre-trained vision models
using HTTP, FileIO, ImageMagick   # load image from URL

# Load model
model = VGG(16; pretrain=true).layers
model = strip_softmax(model)
model = canonize(model)

# Load input
url = HTTP.URI("https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = load(url)
input = preprocess_imagenet(img)
input = reshape(input, 224, 224, 3, :)  # reshape to WHCN format

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
