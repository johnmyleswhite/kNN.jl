export NaiveNeighborTree, nearest

immutable NaiveNeighborTree <: NearestNeighborTree
    X::Matrix
    metric::Metric
end

function nearest(t::NaiveNeighborTree, x::Vector, k::Int)	
	D = colwise(t.metric, x, t.X)
	I = sortperm(D)[1:k]
	return I, D[I]
end