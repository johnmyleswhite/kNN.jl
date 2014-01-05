immutable kNNClassifier
    t::NaiveNeighborTree
    y::Vector
end

function knn(X::Matrix,
             y::Vector;
             metric::Metric = Euclidean())
    return kNNClassifier(NaiveNeighborTree(X, metric), y)
end

# TODO: Don't construct copy of model.y just to extract majority vote
function Stats.predict(model::kNNClassifier,
                       x::Vector,
                       k::Integer = 1)
    inds, dists = nearest(model.t, x, k)
    return majority_vote(model.y[inds])
end

function Stats.predict!(predictions::Vector,
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

function Stats.predict(model::kNNClassifier,
                       X::Matrix,
                       k::Integer = 1)
    predictions = Array(eltype(model.y), size(X, 2))
    predict!(predictions, model, X, k)
    return predictions
end
