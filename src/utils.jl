"""
    stabilize_denom(d; eps = 1f-6)

Replace zero terms of a matrix `d` with `eps`.
"""
function stabilize_denom(d; eps=1f-9)
    return d + eps * (d .≈ 0)
end

"""
    safedivide(a, b; eps = 1f-6)

Elementwise division of two matrices avoiding zero terms
in the denominator by replacing them with `eps`.
"""
function safedivide(a, b; eps=1f-9)
    return a ./ stabilize_denom(b; eps=eps)
end

"""
    drop_singleton_dims(a)

Drop dimensions of size 1 from array.
"""
function drop_singleton_dims(a::AbstractArray)
    return dropdims(a; dims=tuple(findall(size(a) .== 1)...))
end
