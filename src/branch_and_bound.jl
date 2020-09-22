abstract type BBsearch end

abstract type BrFS <: BBsearch end
abstract type DFS <: BBsearch end
abstract type BFS <: BBsearch end

Priority(::BrFS, f::Function, x::IntervalBox) = diam(x)
Priority(::DFS, f::Function, x::IntervalBox) = minimum(diam.(x))
Priority(::BFS, f::Function, x::IntervalBox) = f(x).lo



function optimize(f::Function, X::IntervalBox; eps::Float64, bisect, ) where {N,T}

  LB = f(X).lo

  init_list = PriorityQueue(X => LB)

 final_list = typeof(X)[ ]

  m = Inf

  iterations = 0

 while length(init_list) > 0

      x = dequeue!(init_list)

     gradient = ForwardDiff.gradient(f, x.v)

      if any(0 .∉ gradient)

          iterations = iterations + 1

          continue

      end

      x_random = random(x, 10)

      x_min, arg_min = findmin([f(a) for a in x_random])

      x_midbox = x_random[arg_min]

      mini = localoptim(f, x_midbox)

      m = min(m, mini)

      if f(x).lo > m

          iterations = iterations + 1

          continue

      end

      if diam(x) < eps

          push!(final_list, x)

          iterations = iterations + 1

          continue

      end

      x1, x2 = bisect(x)

      enqueue!(init_list, x1, f(x1).lo)

      enqueue!(init_list, x2, f(x2).lo)

      iterations = iterations + 1

  end

  final_list = clean_list(f, final_list, m)

  return m, length(final_list), iterations

end
