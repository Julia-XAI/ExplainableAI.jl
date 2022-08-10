using ExplainableAI
using Flux
using Metalhead
using FileIO

# Load model
model = VGG(16; pretrain=true).layers
model = strip_softmax(flatten_chain(model))

# Load input
img = load("./assets/heatmaps/castle.jpg")
input = preprocess_imagenet(img)
input = reshape(input, 224, 224, 3, :)  # reshape to WHCN format

# Run XAI methods
methods = Dict(
    "InputTimesGradient" => InputTimesGradient,
    "Gradient" => Gradient,
    "SmoothGrad" => SmoothGrad,
    "IntegratedGradients" => IntegratedGradients,
    "LRP" => LRP,
    "LRPEpsilonGammaBox" => m -> LRP(m, EpsilonGammaBox(-3.0f0, 3.0f0)),
    # "LRPEpsilonPlus" => m -> LRP(m, EpsilonPlus()),
    # "LRPEpsilonAlpha2Beta1" => m -> LRP(m, EpsilonAlpha2Beta1()),
    # "LRPEpsilonPlusFlat" => m -> LRP(m, EpsilonPlusFlat()),
    # "LRPEpsilonAlpha2Beta1Flat" => m -> LRP(m, EpsilonAlpha2Beta1Flat()),
)

for (name, method) in methods
    analyzer = method(model)
    h = heatmap(input, analyzer)
    save("castle_$name.png", h)

    # Output neuron 920 corresponds to "street sign"
    h = heatmap(input, analyzer, 920)
    save("streetsign_$name.png", h)
end
