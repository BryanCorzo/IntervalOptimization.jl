module IntervalOptimization

using IntervalArithmetic
using ForwardDiff

const gradient = ForwardDiff.gradient
const ∇ = gradient

include("basic_functions.jl")

end
