module TestKDTree
	using Base.Test
	using Distance
	using kNN
	using StatsBase

	# Empty tree
	t = KDTree{Vector}()
	@test typeof(t.root) == kNN.EmptyKDTree{Vector}

	t = KDTree{Vector}(Euclidean())
	@test typeof(t.root) <: kNN.EmptyKDTree
	@test typeof(t.metric) == Euclidean

	# With one element
	x = [1, 2, 3]
	t = KDTree{Vector{Int}}(x, 1, Euclidean())
	@test typeof(t.root) <: kNN.KDTreeNode
	@test t.root.k == x
	@test t.root.i == 1

	# With multiple points
	X = [1  60  29  7 86  44 23 54 12]
	t = KDTree{Vector{Int}}(X, Euclidean())
	@test typeof(t.root) <: kNN.KDTreeNode

	# Search
	Distance.evaluate(m::Metric, a::Int, b::Float64) = evaluate(m, float(a) ,b)
	ind, dist = nearest(t, [3], 2)
	@test length(ind) == 2
	@test length(dist) ==  2
	@test 4 in ind && 1 in ind
	@test_approx_eq(4.0, dist[1])
	@test_approx_eq(2.0, dist[2])
end