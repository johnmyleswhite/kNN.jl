# Return the majority of the k-nearest neighbors
function knn(train_features::Matrix,
             test_features::Matrix,
             train_labels::Vector,
             k::Integer)
    n_train = size(train_features, 1)
    n_test = size(test_features, 1)

    D = distances(vcat(train_features, test_features))
    # Search training cases for neighbors of test cases
    D = D[(n_train + 1):(n_train + n_test), 1:n_train]

    predictions = Array(eltype(train_labels), n_test)

    for i in 1:n_test
        neighbors = k_nearest_neighbors(k, i, D)
        predictions[i] = majority_vote(train_labels[neighbors])
    end

    return predictions
end

function knn(train_features::DataFrame,
             test_features::Matrix,
             train_labels::Vector,
             k::Integer)
    knn(matrix(train_features), test_features, train_labels, k)
end

function knn(train_features::DataFrame,
             test_features::DataFrame,
             train_labels::Vector,
             k::Integer)
    knn(matrix(train_features), matrix(test_features), train_labels, k)
end

function knn(train_features::Matrix,
             test_features::DataFrame,
             train_labels::Vector,
             k::Integer)
    knn(train_features, matrix(test_features), train_labels, k)
end

# TODO: Conversion of AbstractVector, etc...

# TODO: Add Formula interface
