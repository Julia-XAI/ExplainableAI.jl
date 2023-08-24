using ExplainableAI
using ExplainableAI: in_branch
using Metalhead
using Flux

model = VGG(11; pretrain=false).layers
model_flat = flatten_model(model)
Flux.testmode!(model, true)
Flux.testmode!(model_flat, true)

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

# Test utilities
@test in_branch((1, 2), 1)
@test !in_branch((1, 2), 2)
@test in_branch((1, 2), (1, 2))
@test in_branch((1, 2, 3), (1, 2))
@test !in_branch((1, 2), (1, 2, 3))

# This composite is non-sensical, but covers many composite primitives
composite1 = Composite(
    ZeroRule(), # default rule
    GlobalMap(PassRule()), # override default rule
    GlobalTypeMap(
        ConvLayer    => AlphaBetaRule(2.0f0, 1.0f0),
        Dense        => EpsilonRule(1.0f-6),
        PoolingLayer => EpsilonRule(1.0f-6),
    ),
    FirstNTypeMap(7, Conv => FlatRule()),
    RangeTypeMap(4:10, PoolingLayer => EpsilonRule(1.0f-5)),
    LayerMap(9, AlphaBetaRule(1.0f0, 0.0f0)),
    FirstLayerMap(ZBoxRule(-3.0f0, 3.0f0)),
    RangeMap(18:19, ZeroRule()),
    LastLayerMap(PassRule()),
)
@test_reference "references/show/composite1.txt" repr("text/plain", composite1)

analyzer1 = LRP(model_flat, composite1; flatten=false)
@test analyzer1.rules == ChainTuple(
    ZBoxRule(-3.0f0, 3.0f0),
    EpsilonRule(1.0f-6),
    FlatRule(),
    EpsilonRule(1.0f-5),
    FlatRule(),
    FlatRule(),
    EpsilonRule(1.0f-5),
    AlphaBetaRule(2.0f0, 1.0f0),
    AlphaBetaRule(1.0f0, 0.0f0),
    EpsilonRule(1.0f-5),
    AlphaBetaRule(2.0f0, 1.0f0),
    AlphaBetaRule(2.0f0, 1.0f0),
    EpsilonRule(1.0f-6),
    PassRule(),
    EpsilonRule(1.0f-6),
    PassRule(),
    EpsilonRule(1.0f-6),
    ZeroRule(),
    PassRule(),
)
@test_reference "references/show/lrp1.txt" repr("text/plain", analyzer1)

model2 = Chain(
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
    LastLayerTypeMap(Dense => EpsilonRule(2.0f-5), Conv => EpsilonRule(2.0f-4)),
    FirstLayerTypeMap(
        Dense => AlphaBetaRule(1.0f0, 0.0f0), Conv => AlphaBetaRule(2.0f0, 1.0f0)
    ),
)
@test_reference "references/show/composite2.txt" repr("text/plain", composite2)

analyzer2 = LRP(model2, composite2; flatten=false)
@test analyzer2.rules == ChainTuple(
    AlphaBetaRule(2.0f0, 1.0f0),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    EpsilonRule(2.0f-5),
)
@test_reference "references/show/lrp2.txt" repr("text/plain", analyzer2)

composite3 = Composite(
    GlobalTypeMap(
        ConvLayer      => ZPlusRule(),
        Dense          => EpsilonRule(),
        DropoutLayer   => PassRule(),
        ReshapingLayer => PassRule(),
    ),
    FirstLayerTypeMap(ConvLayer => FlatRule(), Dense => FlatRule()),
    LastLayerMap(EpsilonRule(1.0f-5)),
)

analyzer3 = LRP(model, composite3; flatten=false)
@test analyzer3.rules == ChainTuple(
    ChainTuple(
        FlatRule(),
        ZeroRule(),
        ZPlusRule(),
        ZeroRule(),
        ZPlusRule(),
        ZPlusRule(),
        ZeroRule(),
        ZPlusRule(),
        ZPlusRule(),
        ZeroRule(),
        ZPlusRule(),
        ZPlusRule(),
        ZeroRule(),
    ),
    ChainTuple(
        PassRule(),
        EpsilonRule(1.0f-6),
        PassRule(),
        EpsilonRule(1.0f-6),
        PassRule(),
        EpsilonRule(1.0f-5),
    ),
)
@test_reference "references/show/lrp3.txt" repr("text/plain", analyzer3)
