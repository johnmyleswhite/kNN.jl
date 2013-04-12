# Find k points closest to x. This excludes x
function k_nearest_neighbors(k::Integer, x::Integer, D::Matrix)
    n = size(D, 2)
    ranked_by_proximity = sortperm(reshape(D[x, :], n))
    res = Array(Int, k)
    index, filled = 1, 1
    while filled <= k
        if ranked_by_proximity[index] == x
            index += 1
            continue
        else
            res[filled] = ranked_by_proximity[index]
            filled += 1
            index += 1
        end
    end
    return res
end
