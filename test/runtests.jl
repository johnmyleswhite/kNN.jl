#
# Correctness Tests
#

using kNN

my_tests = ["majority_vote.jl",
            "classifier.jl",
            "regress.jl",
            "kdtree.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
