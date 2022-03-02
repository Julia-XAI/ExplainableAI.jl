using ExplainabilityMethods

# Flux layers
unknown_function(x) = x
@test check_model(Chain(Dense(2, 2, relu)))
@test (@test_logs (:warn, "Unknown or unsupported activation functions found in model") check_model(
    Chain(Dense(2, 2, softmax)); verbose=false
)) == false
@test (@test_logs (:warn, "Unknown layers found in model") check_model(
    Chain(unknown_function); verbose=false
)) == false

# Custom layers
## Test using a simple wrapper
struct MyLayer{T}
    x::T
end
TestLayer = MyLayer(Dense(2, 2, relu))
@test (@test_logs (:warn, "Unknown layers found in model") check_model(
    Chain(TestLayer); verbose=false
)) == false
## Test should pass after registering the layer
LRP_CONFIG.supports_layer(::MyLayer) = true
@test check_model(Chain(TestLayer); verbose=false) == true
## ...repeat for layers that are functions
@test (@test_logs (:warn, "Unknown layers found in model") check_model(
    Chain(unknown_function); verbose=false
)) == false
LRP_CONFIG.supports_layer(::typeof(unknown_function)) = true
@test check_model(Chain(unknown_function); verbose=false) == true
## ...repeat for activation functions
@test (@test_logs (:warn, "Unknown or unsupported activation functions found in model") check_model(
    Chain(Dense(2, 2, unknown_function)); verbose=false
)) == false
LRP_CONFIG.supports_activation(::typeof(unknown_function)) = true
@test check_model(Chain(Dense(2, 2, unknown_function)); verbose=false) == true
