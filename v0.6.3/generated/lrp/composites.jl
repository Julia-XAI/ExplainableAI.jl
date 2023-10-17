using ExplainableAI
using Flux

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1),
        Conv((3, 3), 16 => 16, relu; pad=1),
        MaxPool((2, 2)),
    ),
    Chain(
        Flux.flatten,
        Dense(1024 => 512, relu),
        Dropout(0.5),
        Dense(512 => 100, relu)
    ),
);

model_flat = flatten_model(model)

rules = [
    FlatRule(),
    ZPlusRule(),
    ZeroRule(),
    ZPlusRule(),
    ZPlusRule(),
    ZeroRule(),
    PassRule(),
    EpsilonRule(),
    PassRule(),
    EpsilonRule(),
];

LRP(model_flat, rules)

rules = ChainTuple(
    ChainTuple(
        FlatRule(),
        ZPlusRule(),
        ZeroRule(),
        ZPlusRule(),
        ZPlusRule(),
        ZeroRule()
    ),
    ChainTuple(
        PassRule(),
        EpsilonRule(),
        PassRule(),
        EpsilonRule(),
    ),
)

analyzer = LRP(model, rules; flatten=false)

composite = Composite(
    GlobalTypeMap( # the following maps of layer types to LRP rules are applied globally
        Conv                 => ZPlusRule(),   # apply ZPlusRule on all Conv layers
        Dense                => EpsilonRule(), # apply EpsilonRule on all Dense layers
        Dropout              => PassRule(),    # apply PassRule on all Dropout layers
        MaxPool              => ZeroRule(),    # apply ZeroRule on all MaxPool layers
        typeof(Flux.flatten) => PassRule(),    # apply PassRule on all flatten layers
    ),
    FirstLayerMap( # the following rule is applied to the first layer
        FlatRule()
    ),
);

analyzer = LRP(model, composite; flatten=false)

lrp_rules(model, composite)

show_layer_indices(model)

composite = Composite(LayerMap((1, 5), EpsilonRule()))

LRP(model, composite; flatten=false)

composite = EpsilonPlusFlat()

analyzer = LRP(model, composite; flatten=false)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
