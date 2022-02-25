"""
    stabilize_denom(d; eps = 1f-6)

Replace zero terms of a matrix `d` with `eps`.
"""
function stabilize_denom(d; eps=1.0f-9)
    return d + ((d .≈ 0) + sign.(d)) * eps
end

"""
    safedivide(a, b; eps = 1f-6)

Elementwise division of two matrices avoiding near zero terms
in the denominator by replacing them with `± eps`.
"""
safedivide(a, b; eps=1.0f-9) = a ./ stabilize_denom(b; eps=eps)

"""
    drop_singleton_dims(a)

Drop dimensions of size 1 from array.
"""
function drop_singleton_dims(a::AbstractArray)
    return dropdims(a; dims=tuple(findall(size(a) .== 1)...))
end
