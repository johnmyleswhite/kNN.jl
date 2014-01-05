using Stats
using Distance

module kNN
    export knn

    using Stats
    using Distance
    using NearestNeighbors

    include("majority_vote.jl")
    include("classifier.jl")
end
