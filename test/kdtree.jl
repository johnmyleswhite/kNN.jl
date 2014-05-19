module TestKDTree
	using Base.Test
	using Distance
	using kNN
	
	# Empty tree
	t = KDTree{Vector,Int}()
	@test typeof(t.root) == kNN.EmptyKDTree{Vector,Int}

	t = KDTree{Vector,Int}(Euclidean())
	@test typeof(t.root) <: kNN.EmptyKDTree
	@test typeof(t.metric) == Euclidean

	# With one element
	x = [1, 2, 3]
	t = KDTree{Vector{Int},Int}(x, 1, Euclidean())
	@test typeof(t.root) == kNN.KDTreeNode{Vector{Int},Int}
	@test t.root.k == x
	@test t.root.v == 1	

	# With multiple points	
	X = [1  60  29  86  44]
	t = KDTree{Vector{Int},Int}(X, Euclidean())
	@test typeof(t.root) <: kNN.KDTreeNode

	# Search
	ind, dist = kNN.nearest(t, [3], 2)
	@test length(ind) == 2
	@test length(dist) ==  2
	@test 3 in ind && 1 in ind
	@test 26.0 in dist && 2.0 in dist
end