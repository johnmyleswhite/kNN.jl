immutable kNNClassifier
    t::NaiveNeighborTree
    y::Vector
end

function knn(X::Matrix,
             y::Vector;
             metric::Metric = Euclidean())
    return kNNClassifier(NaiveNeighborTree(X, metric), y)
end

function Stats.predict(model::kNNClassifier,
                       X::Matrix,
                       k::Integer = 1)
    n = size(X, 2)
    predictions = Array(eltype(model.y), n)
    for i in 1:n
        inds, dists = nearest(model.t, X[:, i], k)
        predictions[i] = majority_vote(model.y[inds])
    end
    return predictions
end
