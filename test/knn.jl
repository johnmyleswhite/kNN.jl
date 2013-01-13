using RDatasets, Resampling

iris = data("datasets", "iris")

train_set, test_set = splitrandom(iris, 2/3)

train_features = matrix(train_set[:, 2:5])
test_features = matrix(test_set[:, 2:5])

train_labels = vector(train_set[:, 6])
test_labels = vector(test_set[:, 6])

for k in [1, 2, 5]
	preds = knn(train_features, test_features, train_labels, k)

	species = unique(train_labels)
	for i in 1:length(preds)
		@assert contains(species, preds[i])
	end

	correct = preds .== test_labels
	accuracy = sum(correct) / length(correct)
	@assert accuracy > 0.33
end

# Same analysis, but with numeric labels

train_labels = int(train_labels .== "setosa")
test_labels = int(test_labels .== "setosa")

preds = knn(train_features, test_features, train_labels, 10)
correct = preds .== test_labels
accuracy = sum(correct) / length(correct)
@assert accuracy > 0.33
