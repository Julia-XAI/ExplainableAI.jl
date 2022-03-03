using ExplainabilityMethods
using Suppressor

# Flux layers
unknown_function(x) = x
@test check_model(:LRP, Chain(Dense(2, 2, relu)))
@test_throws ArgumentError("Unknown or unsupported activation functions found in model") check_model(
    :LRP, Chain(Dense(2, 2, softmax)); verbose=false
)
@test_throws ArgumentError("Unknown layers found in model") check_model(
    :LRP, Chain(unknown_function); verbose=false
)
@test_throws ArgumentError("Unknown layers found in model") @suppress check_model(
    :LRP,
    Chain(
        unknown_function,
        Chain(unknown_function),
        Parallel(+, unknown_function, unknown_function),
    );
    verbose=false,
)

# Custom layers
## Test using a simple wrapper
struct MyLayer{T}
    x::T
end
TestLayer = MyLayer(Dense(2, 2, relu))
@test_throws ArgumentError("Unknown layers found in model") check_model(
    :LRP, Chain(TestLayer); verbose=false
)
@test_throws ArgumentError("Unknown layers found in model") LRPZero(
    Chain(TestLayer); verbose=false
)
@test_nowarn LRPZero(Chain(TestLayer); skip_checks=true)
## Test should pass after registering the layer
LRP_CONFIG.supports_layer(::MyLayer) = true
@test check_model(:LRP, Chain(TestLayer); verbose=false) == true
@test_nowarn LRPZero(Chain(TestLayer))
## ...repeat for layers that are functions
@test_throws ArgumentError("Unknown layers found in model") check_model(
    :LRP, Chain(unknown_function); verbose=false
)
LRP_CONFIG.supports_layer(::typeof(unknown_function)) = true
@test check_model(:LRP, Chain(unknown_function); verbose=false) == true
## ...repeat for activation functions
@test_throws ArgumentError("Unknown or unsupported activation functions found in model") check_model(
    :LRP, Chain(Dense(2, 2, unknown_function)); verbose=false
)
LRP_CONFIG.supports_activation(::typeof(unknown_function)) = true
@test check_model(:LRP, Chain(Dense(2, 2, unknown_function)); verbose=false) == true
