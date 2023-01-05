immutable kNNClassifier
    t::BruteTree
    y::Vector
end

function knn(X::Matrix,
             y::Vector;
             metric::Metric = Euclidean())
    return kNNClassifier(BruteTree(X, metric), y)
end

# TODO: Don't construct copy of model.y just to extract majority vote
function StatsBase.predict(model::kNNClassifier,
                           x::Vector,
                           k::Integer = 1)
    inds, dists = NearestNeighbors.knn(model.t, x, k)
    return majority_vote(model.y[inds])
end

function StatsBase.predict!(predictions::Vector,
                            model::kNNClassifier,
                            X::Matrix,
                            k::Integer = 1)
    n = size(X, 2)
    # @assert eltype(predictions) == eltype(model.y)
    # @assert length(predictions) == n
    for i in 1:n
        predictions[i] = predict(model, X[:, i], k)
    end
    return predictions
end

function StatsBase.predict(model::kNNClassifier,
                           X::Matrix,
                           k::Integer = 1)
    predictions = Array{eltype(model.y)}(size(X, 2))
    predict!(predictions, model, X, k)
    return predictions
end
