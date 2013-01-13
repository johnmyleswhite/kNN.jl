kNN.jl
======

Basic k-nearest neighbors classification. See [Wikipedia](http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm) for a description of the k-nearest neighbors algorithm.

# API

Use the `knn()` function as follows:

    knn(labelled_features, unlabelled_features, labels, k)

The arguments here are:

* `labelled_features`: The features used to determine nearest neighbors for the in-sample data for which the correct classification of observations is already known.
* `unlabelled_features`: The features used to determine nearest neighbors for the out-of-sample data for which the correct classification of observations is unknown.
* `labels`: The correct classification of in-sample data.
* `k`: The number of nearest neighbors used to predict classifications of out-of-sample data.

# Usage

Here we test the `knn()` function on data for which we know the correct classification to assess the capability of the algorithm to impute classes correctly:

    using RDatasets, kNN, Resampling

    iris = data("datasets", "iris")

    train_set, test_set = splitrandom(iris, 2/3)

    train_features = matrix(train_set[:, 2:5])
    test_features = matrix(test_set[:, 2:5])

    train_labels = vector(train_set[:, 6])
    test_labels = vector(test_set[:, 6])

	k = 2
	predictions = knn(train_features, test_features, train_labels, k)

	correct = predictions .== test_labels
	accuracy = sum(correct) / length(correct)
