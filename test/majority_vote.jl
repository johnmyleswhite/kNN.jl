# Tie-breaker selects earliest option for now
# TODO: Debate alternative tie-reconcilation mechanisms
@assert isequal(kNN.majority_vote(["A", "B", "B", "A", "C", "B"]), "B")
@assert isequal(kNN.majority_vote(["A", "B", "B", "A", "C", "B", "C", "C"]), "B")
@assert isequal(kNN.majority_vote(["A", "B", "B", "A", "C", "B", "C", "C", "C"]), "C")
