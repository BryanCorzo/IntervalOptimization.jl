"""`meanvalueform` performs the second order extension on the given function `f` with its optional derivative `f_prime` and initial intervals `x` and  its midpoints `x_mid`."""

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




"""`smear` performs the election of the variable with the largest value of semar like direction to bisect on the given function `f` with its optional derivative `f_prime` and initial intervals `x`."""

function smear(f::Function, f_prime::Function, x::IntervalBox)

    smear_magnitude = abs.(f_prime(x)) .* diam.(X)

    return argmax([X.hi for X in ub])

end


# use automatic differentiation if no derivative function given:

smear(f::Function, x::IntervalBox) = smear(f, x->∇(f,x.v), x)
