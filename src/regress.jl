immutable KernelRegression{R <: Union(Vector, Matrix), T <: Real}
    x::R
    y::Vector{T}
    k::Function
    h::Float64
end

# Want a version for x as a vector (this) and X as a matrix (doesn't exist yet)
function kernelregression{R <: Any,
                          T <: Real}(x::Vector{R},
                                     y::Vector{T};
                                     kernel::Symbol = :epanechnikov,
                                     bandwidth::Real = NaN,
                                     getbandwidth::Function = StatsBase.bandwidth)
    if isnan(bandwidth)
        h_x = getbandwidth(x)
        h_y = getbandwidth(y)
        h = sqrt(h_x * h_y)
    else
        h = bandwidth
    end
    k = SmoothingKernels.kernels[kernel]
    return KernelRegression(x, y, k, h)
end

function StatsBase.predict(model::KernelRegression, x::Real)
    y, n = 0.0, length(model.x)
    h, k = model.h, model.k
    normalizer = 0.0
    for i in 1:n
        # TODO: Get definitions and usage of bandwidth right
        # w_i = (1 / h) * k((x - model.x[i]) / h)
        w_i = h * k(h * (x - model.x[i]))
        y += model.y[i] * w_i
        normalizer += w_i
    end
    return y / normalizer
end

function StatsBase.predict!{T <: Real}(ys::Vector,
                                       model::KernelRegression,
                                       xs::AbstractVector{T})
    n = length(xs)
    # @assert length(ys) == n
    for i in 1:n
        ys[i] = predict(model, xs[i])
    end
    return
end

function StatsBase.predict{T <: Real}(model::KernelRegression,
                                      xs::AbstractVector{T})
    ys = Array(Float64, length(xs))
    predict!(ys, model, xs)
    return ys
end
