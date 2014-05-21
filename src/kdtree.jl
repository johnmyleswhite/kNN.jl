import Base: setindex!, show

using Base.Collections

export KDTree, nearest

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
		next_sd = ((n.d) % length(k))+1

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
	stack = Array(KDTreeNode{K, V}, k)
	h = Float64[]
	index = Dict{Float64,V}()
	sti = 1
	stack[sti] = t.root
	T = 1.0e10
	while sti > 0 # length(stack) > 0 
		n = stack[sti] # n = pop!(stack)		
		sti -= 1
		if n.leaf
			d = evaluate(t.metric, x, n.k)			
			if (length(h) < k) || (d < T)
				while length(h) >= k
					p = heappop!(h, Base.Order.Reverse) 				
					pop!(index, p)
				end
				heappush!(h, d, Base.Order.Reverse)
				push!(index, d, n.v)
				T =  h[1]								
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
			d = (x[n.d]-n.s)*(x[n.d]-n.s)		
			if length(h) < k || d <= T #evaluate(t.metric, x[n.d], n.s) <= T				
				sti += 1
				if sti > length(stack)
					push!(stack, far)
				else
					stack[sti] = far
				end	
				# push!(stack, far)			
			end

			# Search the nearest branch
			sti += 1
			if sti > length(stack)
				push!(stack, near)
			else
				stack[sti] = near
			end	
			#push!(stack, near)		
		end
	end

	vals = Array(eltype(index)[2], k)
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
