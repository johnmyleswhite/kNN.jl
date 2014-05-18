using StatsBase
using Distance

module kNN
    export knn, kernelregression

    using StatsBase
    using Distance    
    using SmoothingKernels

    include("majority_vote.jl")
    include("classifier.jl")
    include("regress.jl")
end
