module IntervalOptimization

using IntervalArithmetic
using ForwardDiff
using StaticArrays
using DataStructures


const gradient = ForwardDiff.gradient
const ∇ = gradient

include("basic_functions.jl")
include("branch_and_bound.jl")


end
