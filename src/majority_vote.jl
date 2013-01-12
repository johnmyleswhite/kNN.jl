# Find the majority vote from k-neighbors
function majority_vote(labels::Vector)
    counts = xtabs(labels)
    res = labels[1]
    max_value = -1
    for (k, v) in counts
        if v > max_value
            res, max_value = k, v
        end
    end
    return res
end
