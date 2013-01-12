#
# Correctness Tests
#

using kNN

my_tests = ["test/majority_vote.jl",
            "test/k_nearest_neighbors.jl",
            "test/knn.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
