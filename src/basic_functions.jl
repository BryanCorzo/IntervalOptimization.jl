"""`meanvalueform` performs the second order extension on the given function `f` with its optional derivative
`f_prime` and initial intervals `x` and  its midpoints `x_mid`."""

function meanvalueform(f::Function, f_prime::Function, x::IntervalBox, x_mid::IntervalBox)

    mvf = f(x_mid) + dot(f_prime(x), (x - x_mid).v)

    return mvf

end


# use automatic differentiation if no derivative function given:

meanvalueform(f::Function, x::IntervalBox, x_mid::IntervalBox) = meanvalueform(f, x->∇(f,x.v), x, x_mid)

# determines the midpoints of x if x_mid is not given:

meanvalueform(f::Function, f_prime, x::IntervalBox) = meanvalueform(f, f_prime, x, IntervalBox(mid(x)))

# use automatic differentiation if no derivative function given and determines the midpoints of x if x_mid is not given:

meanvalueform(f::Function, x::IntervalBox) = meanvalueform(f, x->∇(f,x.v), x, IntervalBox(mid(x)))




"""`smear` performs the election of the variable with the largest value of semar like direction to bisect on the given function `f`
with its optional derivative `f_prime` and initial intervals `x`."""

function smear(f::Function, f_prime::Function, x::IntervalBox)

    smear_magnitude = abs.(f_prime(x)) .* diam.(X)

    return argmax([X.hi for X in ub])

end


# use automatic differentiation if no derivative function given:

smear(f::Function, x::IntervalBox) = smear(f, x->∇(f,x.v), x)


"""`localoptim` performs a local optimization of a given function `f` on intervals `x`."""

function localoptim(f::Function, X::StaticArray)

    xc = X
    gamma = 0.01
    max_iters = 10

    for i in 1:max_iters

        x = xc
        xc = x - gamma * ∇(f, x)

        if f(xc) > f(x)

            break

        end

    end

    return (f(xc), xc)

end


"""`random` generates random numbers over intervals `x`."""

randnum(r::SArray, x::IntervalBox) = (r * (x.hi - x.lo)) + x.lo

function random(X::IntervalBox{N,T}, n::Int64) where {N,T}

    Random_numbers = typeof(mid(X))[ ]

    for i in 1:n

        rand_numb = @SVector rand(N)

        interval_rand_numb = randnum.(rand_numb, X)

        push!(Random_numbers, interval_rand_numb)

    end

    return Random_numbers

end
