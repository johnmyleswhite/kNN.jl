D = [0.0 1.0 2.0;
     1.0 0.0 3.0;
     2.0 3.0 0.0;]

@assert isequal(kNN.k_nearest_neighbors(1, 1, D), [2])
@assert isequal(kNN.k_nearest_neighbors(1, 2, D), [1])
@assert isequal(kNN.k_nearest_neighbors(1, 3, D), [1])

@assert isequal(kNN.k_nearest_neighbors(2, 1, D), [2, 3])
@assert isequal(kNN.k_nearest_neighbors(2, 2, D), [1, 3])
@assert isequal(kNN.k_nearest_neighbors(2, 3, D), [1, 2])

D = [0.0 2.0 1.0;
     2.0 0.0 0.5;
     1.0 0.5 0.0;]

@assert isequal(kNN.k_nearest_neighbors(1, 1, D), [3])
@assert isequal(kNN.k_nearest_neighbors(1, 2, D), [3])
@assert isequal(kNN.k_nearest_neighbors(1, 3, D), [2])

@assert isequal(kNN.k_nearest_neighbors(2, 1, D), [3, 2])
@assert isequal(kNN.k_nearest_neighbors(2, 2, D), [3, 1])
@assert isequal(kNN.k_nearest_neighbors(2, 3, D), [2, 1])
