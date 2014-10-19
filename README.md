kNN.jl
======

Basic k-nearest neighbors classification and regression.

For a list of the distance metrics that can be used in k-NN classification, see [Distances.jl](https://github.com/JuliaStats/Distances.jl)

For a list of the smoothing kernels that can be used in kernel regression, see [SmoothingKernel.jl](https://github.com/johnmyleswhite/SmoothingKernels.jl)

# Example 1: Classification using Euclidean distance

    using kNN
    using DataArrays
    using DataFrames
    using RDatasets
    using Distances

    iris = data("datasets", "iris")
    X = array(iris[:, 1:4])'
    y = array(iris[:, 5])
    model = knn(X, y, metric = Euclidean())

    predict_k1 = predict(model, X, 1)
    predict_k2 = predict(model, X, 2)
    predict_k3 = predict(model, X, 3)
    predict_k4 = predict(model, X, 4)
    predict_k5 = predict(model, X, 5)

    mean(predict_k1 .== y)
    mean(predict_k2 .== y)
    mean(predict_k3 .== y)
    mean(predict_k4 .== y)
    mean(predict_k5 .== y)

# Example 2: Classification using Manhattan distance

    using kNN
    using DataArrays
    using DataFrames
    using RDatasets
    using Distances

    iris = data("datasets", "iris")
    X = array(iris[:, 1:4])'
    y = array(iris[:, 5])
    model = knn(X, y, metric = Cityblock())

    predict_k1 = predict(model, X, 1)
    predict_k2 = predict(model, X, 2)
    predict_k3 = predict(model, X, 3)
    predict_k4 = predict(model, X, 4)
    predict_k5 = predict(model, X, 5)

    mean(predict_k1 .== y)
    mean(predict_k2 .== y)
    mean(predict_k3 .== y)
    mean(predict_k4 .== y)
    mean(predict_k5 .== y)

# Example 3: Regression using Gaussian kernel

    using Base.Test
    using kNN
    using StatsBase

    srand(1)
    n = 1_000
    x = 10 * randn(n)
    y = sin(x) + 0.5 * randn(n)

    fit = kernelregression(x, y, kernel = :gaussian)
    grid = minimum(x):0.1:maximum(x)
    predictions = predict(fit, grid)

# To Do

* Allow user to request that `knn` generate a ball tree, KD-tree or cover tree as a method for conducting nearest neighbor searches.
* Allow user to request that approximate nearest neighbors be returned instead of exact nearest neighbors.
* Clean up API
