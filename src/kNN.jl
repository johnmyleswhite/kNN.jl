require("DataFrames")
using DataFrames

module kNN
    export knn

    using DataFrames

    include("majority_vote.jl")
    include("k_nearest_neighbors.jl")
    include("classify.jl")
end
