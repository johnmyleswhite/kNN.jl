using StatsBase
using Distances

module kNN
    export knn, kernelregression

    using StatsBase
    using Distances
    using NearestNeighbors
	using SmoothingKernels

    include("majority_vote.jl")
    include("classifier.jl")
    include("regress.jl")
end
