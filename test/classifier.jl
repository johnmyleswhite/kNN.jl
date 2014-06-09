module TestClassifier
	using Base.Test
    using Distance
    using NearestNeighbors
    using kNN

    iris, _ = readdlm(Pkg.dir("kNN", "test", "data", "iris.csv"), ',', header=true)
    X = float64(iris[:, 1:4])'
    y = iris[:, 5]

    model = knn(X, y) #NaiveNeighborTree
    predict_k1 = predict(model, X, 1)
    predict_k2 = predict(model, X, 2)
    predict_k3 = predict(model, X, 3)
    predict_k4 = predict(model, X, 4)
    predict_k5 = predict(model, X, 5)

    @test sum(predict_k1 .== y) == 150
    @test sum(predict_k2 .== y) == 147
    @test sum(predict_k3 .== y) == 144
    @test sum(predict_k4 .== y) == 145
    @test sum(predict_k5 .== y) == 145

    model = knn(X, y, KDTree)
    predict_k1 = predict(model, X, 1)
    predict_k2 = predict(model, X, 2)
    predict_k3 = predict(model, X, 3)
    predict_k4 = predict(model, X, 4)
    predict_k5 = predict(model, X, 5)

    @test sum(predict_k1 .== y) == 150
    @test sum(predict_k2 .== y) == 147
    @test sum(predict_k3 .== y) == 144
    @test sum(predict_k4 .== y) == 145
    @test sum(predict_k5 .== y) == 145
end
