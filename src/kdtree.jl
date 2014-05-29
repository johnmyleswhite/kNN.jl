import Base: setindex!, show

using Base.Collections
using Distance

export KDTree, nearest

abstract AbstractKDTree{K} <: AbstractNearestNeighborTree

type EmptyKDTree{K} <: AbstractKDTree{K}
end

type KDTreeNode{K} <: AbstractKDTree{K}
    k::     K               # multidimensional point
    i::		Int             # index
    d::     Int             # split dimension
    s::     Float64         # split value
    leaf::  Bool            # is leaf?
    left::  AbstractKDTree  # left node
    right:: AbstractKDTree  # right node

    KDTreeNode(k::K, i::Int) = new(k, i, 1, NaN, true, EmptyKDTree{K}(), EmptyKDTree{K}())
    KDTreeNode(k::K, i::Int, d) = new(k, i, d, NaN, true, EmptyKDTree{K}(), EmptyKDTree{K}())
end

type KDTree{K} <: NearestNeighborTree
	metric::Metric
	root:: AbstractKDTree{K}

	KDTree() = new (Euclidean(), EmptyKDTree{K}())
	KDTree(metric::Metric) = new(metric, EmptyKDTree{K}())
	KDTree(k, i, metric::Metric) = new(metric, setindex!(EmptyKDTree{K}(), i, k))

	function KDTree(X::Matrix, metric::Metric)
		n = size(X, 2)
		t = KDTree{K}(X[:,1], 1, metric)
		for i = 2 : n
			setindex!(t.root, i, X[:,i])
		end
		new(t.metric, t.root)
	end
end

function show(io::IO, t::KDTree)
	println(io, "KDTree: ", t.metric)
end

setindex!{K}(t::EmptyKDTree{K}, i::Int, k) = KDTreeNode{K}(k, i)
setindex!(t::KDTree, i::Int, k) = (t.root = setindex!(t.root, i, k); t)

function setindex!{K}(n::KDTreeNode{K}, i::Int, k::K)
	if n.leaf
		left_k = k
		left_i = i
		right_k = n.k
		right_i = n.i
		if right_k[n.d] < left_k[n.d]
			left_k = n.k
			left_i = n.i
			right_k = k
			right_i = i
		end
		n.s = 0.5*(right_k[n.d] + left_k[n.d])
		next_sd = ((n.d) % length(k))+1

		n.left = KDTreeNode{K}(left_k, left_i, next_sd)
		n.right = KDTreeNode{K}(right_k, right_i, next_sd)
		n.leaf = false
	else
		setindex!(k[n.d] <= n.s ? n.left : n.right, i, k)
	end
	return n
end

disp(t::KDTree) = disp(t.root)
function disp(n::KDTreeNode, l::Int = 0)
	print(repeat("\t", l), "L: $(l), SD: $(n.d), SV: $(n.s)")

	if typeof(n.left) <: EmptyKDTree && typeof(n.right) <: EmptyKDTree
		println(" = K: $(n.k) ($(n.i))")
	else
		println("")
		disp(n.right, l+1)
		disp(n.left, l+1)
	end
end

function search_leaf{K}(n::KDTreeNode{K},
	x::K,
	k::Int,
	h::Array{Float64,1},
	index::Dict{Float64,Int},
	m::Metric)

	if n.leaf
		d = evaluate(m, x, n.k)
		if (length(h) < k) || (d < h[1])
			while length(h) >= k
				p = heappop!(h, Base.Order.Reverse)
				pop!(index, p)
			end
			heappush!(h, d, Base.Order.Reverse)
			push!(index, d, n.i)
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

		# Search the nearest branch
		search_leaf(near, x, k, h, index, m)

		# Only search far tree if do not have enough neighbors
		d = evaluate(m, x[n.d], n.s)
		if length(h) < k || d <= h[1]
			search_leaf(far, x, k, h, index, m)
		end
	end
end

function nearest{K}(t::KDTree{K}, x::K, k::Int)
	h = [Inf]
	index = Dict{Float64,Int}(Inf,1)

	search_leaf(t.root, x, k, h, index, t.metric)

	vals = Array(Int, k)
	dists = Array(Float64, k)
	i = start(index)
	j = 1
	while !done(index, i)
		(v, i) = next(index, i)
		dists[j] = v[1]
		vals[j] = v[2]
		j += 1
	end
	return vals, dists
end

# Distances redefiend
Distance.evaluate{T1 <: FloatingPoint, T2 <: FloatingPoint}(dist::Euclidean, a::T1, b::T2) = (a-b)*(a-b)
