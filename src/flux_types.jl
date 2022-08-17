## Layer types
"""Union type for convolutional layers."""
const ConvLayer = Union{Conv, ConvTranspose, CrossCor}

"""Union type for dropout layers."""
const DropoutLayer = Union{Dropout,typeof(Flux.dropout),AlphaDropout}

"""Union type for reshaping layers such as `flatten`."""
const ReshapingLayer = Union{typeof(Flux.flatten),typeof(Flux.MLUtils.flatten)}

"""Union type for max pooling layers."""
const MaxPoolLayer = Union{MaxPool,AdaptiveMaxPool,GlobalMaxPool}

"""Union type for mean pooling layers."""
const MeanPoolLayer = Union{MeanPool,AdaptiveMeanPool,GlobalMeanPool}

"""Union type for pooling layers."""
const PoolingLayer = Union{MaxPoolLayer,MeanPoolLayer}

# Activation functions
"""Union type for ReLU-like activation functions."""
const ReluLikeActivation = Union{typeof(relu),typeof(gelu),typeof(swish),typeof(mish)}

"""Union type for softmax activation functions."""
const SoftmaxActivation = Union{typeof(softmax),typeof(softmax!)}

# Layers & activation functions supported by LRP
"""Union type for layers that are allowed by default in "deep rectifier networks"."""
const LRPSupportedLayer = Union{Dense,ConvLayer,DropoutLayer,ReshapingLayer,PoolingLayer}

"""Union type for activation functions that are allowed by default in "deep rectifier networks"."""
const LRPSupportedActivation = Union{typeof(identity),ReluLikeActivation}
