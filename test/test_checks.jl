using ExplainabilityMethods

UnknownLayer(x) = x
@test check_model(Chain(Dense(2, 2, relu)))
@test (@test_logs (:warn, "Unknown or unsupported activation functions found in model") check_model(
    Chain(Dense(2, 2, softmax)); verbose=false
)) == false
@test (@test_logs (:warn, "Unknown layers found in model") check_model(
    Chain(UnknownLayer); verbose=false
)) == false
