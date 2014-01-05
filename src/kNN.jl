using Stats
using Distance

module kNN
    export knn, kernelregression

    using Stats
    using Distance
    using NearestNeighbors
	using SmoothingKernels

    include("majority_vote.jl")
    include("classifier.jl")
    include("regress.jl")
end
