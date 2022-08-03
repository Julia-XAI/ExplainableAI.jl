using ExplainableAI
using ImageCore

A = RGB{Float32}[
    RGB{Float32}(0.44557732f0, 0.25328094f0, 0.53720146f0) RGB{Float32}(0.99433f0, 0.37066674f0, 0.8781263f0) RGB{Float32}(0.59815156f0, 0.21008879f0, 0.07259983f0)
    RGB{Float32}(0.6966612f0, 0.27341717f0, 0.40360665f0) RGB{Float32}(0.12119287f0, 0.63196003f0, 0.32167268f0) RGB{Float32}(0.31825548f0, 0.7599565f0, 0.20566207f0)
]
@test_reference "references/preprocess_imagnet" preprocess_imagenet(A)
