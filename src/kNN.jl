using DataFrames

module kNN
    export knn

    using DataFrames, Stats

    include("majority_vote.jl")
    include("k_nearest_neighbors.jl")
    include("classify.jl")
end
