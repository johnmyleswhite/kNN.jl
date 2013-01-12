require("DataFrames")
using DataFrames

module kNN
    export knn

    using DataFrames

    include(joinpath(julia_pkgdir(), "kNN", "src", "majority_vote.jl"))
    include(joinpath(julia_pkgdir(), "kNN", "src", "k_nearest_neighbors.jl"))
    include(joinpath(julia_pkgdir(), "kNN", "src", "classify.jl"))
end
