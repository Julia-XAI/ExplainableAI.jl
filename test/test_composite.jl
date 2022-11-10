using ExplainableAI
using ExplainableAI: rules
using Flux

# Load VGG model:
# We run the reference test on the randomly intialized weights
# so we don't have to download ~550 MB on every CI run.
include("./vgg11.jl")
model = VGG11(; pretrain=false)
model = flatten_model(strip_softmax(model.layers))

# Test default composites
const DEFAULT_COMPOSITES = Dict(
    "EpsilonGammaBox"        => EpsilonGammaBox(-3.0f0, 3.0f0),
    "EpsilonPlus"            => EpsilonPlus(),
    "EpsilonAlpha2Beta1"     => EpsilonAlpha2Beta1(),
    "EpsilonPlusFlat"        => EpsilonPlusFlat(),
    "EpsilonAlpha2Beta1Flat" => EpsilonAlpha2Beta1Flat(),
)
for (name, c) in DEFAULT_COMPOSITES
    @test_reference "references/show/$name.txt" repr("text/plain", c)
end

# This composite is non-sensical, but covers many composite primitives
composite1 = Composite(
    ZeroRule(), # default rule
    GlobalRule(PassRule()), # override default rule
    GlobalTypeRule(
        ConvLayer    => AlphaBetaRule(2.0f0, 1.0f0),
        Dense        => EpsilonRule(1.0f-6),
        PoolingLayer => EpsilonRule(1.0f-6),
    ),
    FirstNTypeRule(7, Conv => FlatRule()),
    LastNTypeRule(3, Dense => EpsilonRule(1.0f-7)),
    RangeTypeRule(4:10, PoolingLayer => EpsilonRule(1.0f-5)),
    LayerRule(9, AlphaBetaRule(1.0f0, 0.0f0)),
    FirstLayerRule(ZBoxRule(-3.0f0, 3.0f0)),
    RangeRule(18:19, ZeroRule()),
    LastLayerRule(PassRule()),
)

analyzer1 = LRP(model, composite1)
@test rules(analyzer1) == [
    ZBoxRule(-3.0f0, 3.0f0)
    EpsilonRule(1.0f-6)
    FlatRule()
    EpsilonRule(1.0f-5)
    FlatRule()
    FlatRule()
    EpsilonRule(1.0f-5)
    AlphaBetaRule(2.0f0, 1.0f0)
    AlphaBetaRule(1.0f0, 0.0f0)
    EpsilonRule(1.0f-5)
    AlphaBetaRule(2.0f0, 1.0f0)
    AlphaBetaRule(2.0f0, 1.0f0)
    EpsilonRule(1.0f-6)
    PassRule()
    EpsilonRule(1.0f-6)
    PassRule()
    EpsilonRule(1.0f-7)
    ZeroRule()
    PassRule()
]
@test_reference "references/show/lrp1.txt" repr("text/plain", analyzer1)
@test_reference "references/show/composite1.txt" repr("text/plain", composite1)

model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
)
composite2 = Composite(
    LastLayerTypeRule(Dense => EpsilonRule(2.0f-5), Conv => EpsilonRule(2.0f-4)),
    FirstLayerTypeRule(
        Dense => AlphaBetaRule(1.0f0, 0.0f0), Conv => AlphaBetaRule(2.0f0, 1.0f0)
    ),
)
analyzer2 = LRP(model, composite2)
@test rules(analyzer2) == [
    AlphaBetaRule(2.0f0, 1.0f0)
    ZeroRule()
    ZeroRule()
    ZeroRule()
    ZeroRule()
    ZeroRule()
    ZeroRule()
    EpsilonRule(2.0f-5)
]
@test_reference "references/show/lrp2.txt" repr("text/plain", analyzer2)
@test_reference "references/show/composite2.txt" repr("text/plain", composite2)
