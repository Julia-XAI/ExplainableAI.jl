using ExplainableAI
using Flux

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16; pad=1),
        BatchNorm(16, relu),
        Conv((3, 3), 16 => 8, relu; pad=1),
        BatchNorm(8, relu),
    ),
    Chain(
        Flux.flatten,
        Dense(2048 => 512, relu),
        Dropout(0.5),
        Dense(512 => 100, softmax)
    ),
);

model = strip_softmax(model)

model = canonize(model)

model_flat = flatten_model(model)

analyzer = LRP(model)
analyzer.model

LRP(model)

composite = EpsilonPlusFlat() # using composite preset EpsilonPlusFlat

LRP(model, composite)

input = rand(Float32, 32, 32, 3, 1) # dummy input for our convolutional neural network

expl = analyze(input, analyzer; layerwise_relevances=true)
expl.extras.layerwise_relevances

analyzer = LRP(model; flatten=false) # use unflattened model

expl = analyze(input, analyzer; layerwise_relevances=true)
expl.extras.layerwise_relevances

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
