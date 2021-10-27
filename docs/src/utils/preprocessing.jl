# Image preprocessing for ImageNet models.
# Code taken from Metalhead 0.5.3's utils.jl
using Images

# Resize an image such that its smallest dimension is the given length
function resize_smallest_dimension(im, len)
    reduction_factor = len / minimum(size(im)[1:2])
    new_size = size(im)
    new_size = (
        round(Int, size(im, 1) * reduction_factor),
        round(Int, size(im, 2) * reduction_factor),
    )
    if reduction_factor < 1.0
        # Images.jl's imresize() needs to first lowpass the image, it won't do it for us
        im = imfilter(im, KernelFactors.gaussian(0.75 / reduction_factor), Inner())
    end
    return imresize(im, new_size)
end

# Take the len-by-len square of pixels at the center of image `im`
function center_crop(im, len)
    l2 = div(len, 2)
    adjust = len % 2 == 0 ? 1 : 0
    return im[
        (div(end, 2) - l2):(div(end, 2) + l2 - adjust),
        (div(end, 2) - l2):(div(end, 2) + l2 - adjust),
    ]
end

function preprocess(im::AbstractMatrix{<:AbstractRGB})
    # Resize such that smallest edge is 256 pixels long
    im = resize_smallest_dimension(im, 256)

    # Center-crop to 224x224
    im = center_crop(im, 224)

    # Convert to channel view and normalize (these coefficients taken
    # from PyTorch's ImageNet normalization code)
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]
    im = (channelview(im) .- μ) ./ σ

    # Convert from CHW (Image.jl's channel ordering) to WHCN (Flux.jl's ordering)
    # and enforce Float32, as that seems important to Flux
    return Float32.(permutedims(im, (3, 2, 1))[:, :, :, :] .* 255)
end
preprocess(im) = preprocess(load(im))
preprocess(im::AbstractMatrix) = preprocess(RGB.(im))
