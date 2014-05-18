using StatsBase
using Distance

module kNN
    export knn, kernelregression

    using StatsBase
    using Distance    
    using SmoothingKernels

    include("majority_vote.jl")
	include("generic.jl")
    include("naive.jl")
    include("classifier.jl")
    include("regress.jl")
end
