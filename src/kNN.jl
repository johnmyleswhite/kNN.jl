module kNN
    export knn, kernelregression, predict

    using StatsBase
    using Distance
    using NearestNeighbors
    using SmoothingKernels

    include("majority_vote.jl")
    include("classifier.jl")
    include("regress.jl")
end
