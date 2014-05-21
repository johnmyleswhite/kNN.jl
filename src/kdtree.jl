import Base: setindex!, show

using Base.Collections
#using DataStructures

export KDTree, nearest, nearest2

abstract AbstractKDTree{K, V} <: AbstractNearestNeighborTree

type EmptyKDTree{K, V} <: AbstractKDTree{K, V}
end

type KDTreeNode{K, V} <: AbstractKDTree{K, V}
    k::     K               # multidimensional point
    v::		V
    d::     Int             # split dimension
    s::     Float64         # split value
    left::  AbstractKDTree  # left node
    right:: AbstractKDTree  # right node
    leaf::  Bool            # is leaf?

    KDTreeNode(k::K, v::V) = new(k, v, 1, NaN, EmptyKDTree{K, V}(), EmptyKDTree{K, V}(), true)
    KDTreeNode(k::K, v::V, d) = new(k, v, d, NaN, EmptyKDTree{K, V}(), EmptyKDTree{K, V}(), true)
end

type KDTree{K, V} <: NearestNeighborTree
	metric::Metric
	root:: AbstractKDTree{K, V}

	KDTree() = new (Euclidean(), EmptyKDTree{K, V}())
	KDTree(metric::Metric) = new(metric, EmptyKDTree{K, V}())
	KDTree(k, v, metric::Metric) = new(metric, setindex!(EmptyKDTree{K, V}(), v, k))

	function KDTree(X::Matrix, metric::Metric)
		n = size(X, 2)
		t = KDTree{K, V}(X[:,1], 1, metric)
		for i = 2 : n
			setindex!(t.root, i, X[:,i])
		end
		new(t.metric, t.root)
	end
end

function show(io::IO, t::KDTree)
	println(io, "KDTree: ", t.metric)
end

setindex!{K,V}(t::EmptyKDTree{K,V}, v, k) = KDTreeNode{K,V}(k, v)
setindex!(t::KDTree, v, k) = (t.root = setindex!(t.root, v, k); t)

function setindex!{K,V}(n::KDTreeNode{K,V}, v::V, k::K)		
	if n.leaf
		left_k = k
		left_v = v
		right_k = n.k		
		right_v = n.v
		if right_k[n.d] < left_k[n.d]
			left_k = n.k
			left_v = n.v
			right_k = k
			right_v = v		
		end
		n.s = 0.5*(right_k[n.d] + left_k[n.d])
		next_sd = ((n.d+1) % length(v))+1

		n.left = KDTreeNode{K,V}(left_k, left_v, next_sd)
		n.right = KDTreeNode{K,V}(right_k, right_v, next_sd)
		n.leaf = false
	else 
		setindex!(k[n.d] <= n.s ? n.left : n.right, v, k)
	end 
	return n
end

disp(t::KDTree) = disp(t.root)
function disp(n::KDTreeNode, l::Int = 0)
	print(repeat("\t", l), "L: $(l), SD: $(n.d), SV: $(n.s)")
	
	if typeof(n.left) <: EmptyKDTree && typeof(n.right) <: EmptyKDTree
		println(" = K: $(n.k), V: $(n.v)")
	else
		println("")
		disp(n.right, l+1)
		disp(n.left, l+1)			
	end
end

function nearest{K, V}(t::KDTree{K, V}, x::K, k::Int)	
	stack = KDTreeNode{K, V}[]
	pq = PriorityQueue{Float64,V}()	
	push!(stack, t.root)
	while length(stack) > 0
		n = pop!(stack)
		if n.leaf
			d = evaluate(t.metric, x, n.k)
			if length(pq) < k || d < peek(pq)[1]
				while length(pq) >= k
					dequeue!(pq)
				end
				enqueue!(pq, d, n.v)
			end
		else
			# Determine nearest and furtherest branch			
			if (x[n.d] > n.s)
				near = n.right
				far = n.left
			else
				near = n.left
				far = n.right
			end

			# Only search far tree if do not have enough neighbors
			if length(pq) < k 
				push!(stack, far)
			end

			# Search the nearest branch
			push!(stack, near)
		end
	end

	vals = eltype(pq)[2][]
	dists = Float64[]
	i = start(pq)
	while !done(pq, i)
		(v, i) = next(pq, i)
		push!(dists, v[1])
		push!(vals, v[2])
	end	
	return vals, dists
end

# function nearest{K, V}(t::KDTree{K, V}, x::K, k::Int)	
# 	stack = KDTreeNode{K, V}[]
# 	h = binary_minheap(Float64)
# 	index = OrderedDict(Float64,V)
# 	push!(stack, t.root)
# 	while length(stack) > 0
# 		n = pop!(stack)
# 		if n.leaf
# 			d = evaluate(t.metric, x, n.k)
# 			if length(h) < k || d < top(h)
# 				while length(h) >= k
# 					p = pop!(h)
# 					pop!(index, p)
# 				end
# 				push!(h, d)
# 				push!(index, d, n.v)
# 			end
# 		else
# 			# Determine nearest and furtherest branch			
# 			if (x[n.d] > n.s)
# 				near = n.right
# 				far = n.left
# 			else
# 				near = n.left
# 				far = n.right
# 			end

# 			# Only search far tree if do not have enough neighbors
# 			if length(h) < k 
# 				push!(stack, far)
# 			end

# 			# Search the nearest branch
# 			push!(stack, near)
# 		end
# 	end

# 	vals = eltype(index)[2][]
# 	dists = Float64[]
# 	i = start(index)
# 	while !done(index, i)
# 		(v, i) = next(index, i)
# 		push!(dists, v[1])
# 		push!(vals, v[2])
# 	end	
# 	return vals, dists
# end