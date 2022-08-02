# Image preprocessing for ImageNet models.
# Code adapted from Metalhead 0.5.3's deprecated utils.jl
# TODO: Remove once matching functionality is in either Metalhead.jl or MLDatasets.jl

# Coefficients taken from PyTorch's ImageNet normalization code
const PYTORCH_MEAN = [0.485f0, 0.456f0, 0.406f0]
const PYTORCH_STD = [0.229f0, 0.224f0, 0.225f0]
const IMGSIZE = (224, 224)

# Take rectangle of pixels of shape `outsize` at the center of image `im`
adjust(i::Integer) = ifelse(iszero(i % 2), 1, 0)
function center_crop_view(im::AbstractMatrix, outsize=IMGSIZE)
    im = imresize(im, ratio=maximum(outsize.// size(im)))
    h2, w2 = div.(outsize, 2) # half height, half width of view
    h_adjust, w_adjust = adjust.(outsize)
    return @view im[
        ((div(end, 2) - h2):(div(end, 2) + h2 - h_adjust)) .+ 1,
        ((div(end, 2) - w2):(div(end, 2) + w2 - w_adjust)) .+ 1,
    ]
end

"""
    preprocess_imagenet(img)

Preprocess an image for use with Metalhead.jl's ImageNet models using PyTorch weights.
Uses PyTorch's normalization constants.
"""
function preprocess_imagenet(im::AbstractMatrix{<:AbstractRGB}, T=Float32::Type{<:Real})
    im = center_crop_view(im)
    im = (channelview(im) .- PYTORCH_MEAN) ./ PYTORCH_STD
    return convert.(T, PermutedDimsArray(im, (3, 2, 1))) # Convert Image.jl's CHW to WHC
end
