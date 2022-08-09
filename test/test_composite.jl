using ExplainableAI
using Flux

# Load VGG model:
# We run the reference test on the randomly intialized weights
# so we don't have to download ~550 MB on every CI run.
include("./vgg11.jl")
model = VGG11(; pretrain=false)
model = flatten_model(strip_softmax(model.layers))

# This composite is non-sensical, but covers as many composite primitives as possible
composite = Composite(
    ZeroRule(), # default rule
    GlobalRule(PassRule()), # override default rule
    GlobalTypeRule(
        ConvLayer => AlphaBetaRule(2.0f0, 1.0f0),
        Dense => EpsilonRule(1.0f-6),
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

analyzer = LRP(model, composite)
@test analyzer.rules == [
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

# Test printing
@test_reference "references/show/composite.txt" repr("text/plain", composite)
# Test default composites
const DEFAULT_COMPOSITES = Dict(
    "EpsilonGammaBox" => EpsilonGammaBox(-3.0f0, 3.0f0),
    "EpsilonPlus" => EpsilonPlus(),
    "EpsilonAlpha2Beta1" => EpsilonAlpha2Beta1(),
    "EpsilonPlusFlat" => EpsilonPlusFlat(),
    "EpsilonAlpha2Beta1Flat" => EpsilonAlpha2Beta1Flat(),
)
for (name, c) in DEFAULT_COMPOSITES
    @test_reference "references/show/$name.txt" repr("text/plain", c)
end
