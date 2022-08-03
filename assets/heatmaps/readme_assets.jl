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
methods = [LRP, InputTimesGradient, Gradient, SmoothGrad, IntegratedGradients]

for method in methods
    analyzer = method(model)
    h = heatmap(input, analyzer)
    save("castle_$method.png", h)

    # Output neuron 920 corresponds to "street sign"
    h = heatmap(input, analyzer, 920)
    save("streetsign_$method.png", h)
end

rules = [
    ZBoxRule(-3.0f0, 3.0f0),
    # FlatRule(),
    FlatRule(),
    EpsilonRule(),
    FlatRule(),
    FlatRule(),
    EpsilonRule(),
    AlphaBetaRule(),
    AlphaBetaRule(),
    AlphaBetaRule(),
    EpsilonRule(),
    GammaRule(),
    GammaRule(),
    GammaRule(),
    EpsilonRule(),
    GammaRule(),
    GammaRule(),
    GammaRule(),
    EpsilonRule(),
    PassRule(),
    EpsilonRule(),
    PassRule(),
    EpsilonRule(),
    PassRule(),
    EpsilonRule(),
]
analyzer = LRP(model, rules)
h = heatmap(input, analyzer)
save("castle_LRP_composite.png", h)
h = heatmap(input, analyzer, 920)
save("streetsign_LRP_composite.png", h)
