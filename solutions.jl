# Based on code written by Zach Sunberg
# https://github.com/zsunberg/OPCSPs.jl

using JuMP
using Gurobi

abstract OPSolver

type GurobiExactSolver <: OPSolver end

# Structure for solution
type OPSolution
    x # vector of fractional vertices to visit (from LP relaxation)
    u # used to remove cycles
end
OPSolution() = OPSolution(nothing,nothing)

# Rounds the LP solution and checks that the path is valid
build_path(op, s::OPSolution) = build_path(op, s.x)

function build_path(op, x)
    xopt = round(Int,x)
    path = [op.start]
    current = -1
    dist_sum = 0.0
    while current != op.stop
        if current==-1
            current=op.start
        end
        # get optimal child
        current = findfirst(xopt[current,:])
        push!(path, current)
    end
    return path
end

# Solves the OP problem.
function solve_op(solver::GurobiExactSolver, op)
    path = build_path(op, gurobi_solve(op))
    @assert distance(op, path) <= op.distance_limit
    return path
end

function solve_exact_op(solver::GurobiExactSolver, op, b_min)
    path = build_path(op, gurobi_solve_exact(op, b_min));
    @assert distance(op,path) <= op.distance_limit
    return path
end

function gurobi_solve_exact(op, bmin; output=0, initial::OPSolution=OPSolution())
    m = Model(solver=Gurobi.GurobiSolver(OutputFlag=output))
    N = length(op) # defined as the number of vertices in the problem.

    without_start = [1:op.start-1; op.start+1:N]
    without_stop = [1:op.stop-1; op.stop+1:N]
    without_both = intersect(without_start, without_stop)

    @defVar(m, x[1:N,1:N], Bin) #NxN array of binary variables - x[i,j] == 1 means j is visited just after i
    @defVar(m, t[1:N]) #Nx1 array of floating point allocations
    @defVar(m, 2 <= u[without_start] <= N, Int) # is u defined here, or elsewhere? u is degree?

    # For hot-starting solver?
    if !is(initial.x, nothing)
        setValue(x, initial.x)
        for keytuple in keys(u)
            key = keytuple[1]
            setValue(u[key], initial.u[key])
        end
    end

# Add square-root here
    @defNLExpr(obj, sum{ sum{op.r[i]*x[i,j]*sqrt(t[i]+bmin), j=1:N}, i=1:N})
    # sum reward of visited nodes
    @setNLObjective(m, Max, obj)#sum{ sum{op.r[i]*x[i,j]*sqrt(t[i] + bmin), j=1:N}, i=1:N })

    @addConstraint(m, min_time[k=1:N], -t[k] <= 0) 
    @addConstraint(m, max_time, sum{t[i]+bmin, i=1:N} <= op.distance_limit)

    # limit one child per parent
    @addConstraint(m, sum{x[op.start,j], j=without_start} == 1)
    @addConstraint(m, sum{x[i,op.stop], i=without_stop} == 1)

    # problem
    @addConstraint(m, connectivity[k=without_both], sum{x[i,k], i=1:N} == sum{x[k,j], j=1:N})

    @addConstraint(m, once[k=1:N], sum{x[k,j], j=1:N} <= 1)
    @addConstraint(m, sum{ sum{op.distances[i,j]*x[i,j], j=1:N}, i=1:N } <= op.distance_limit)
    @addConstraint(m, nosubtour[i=without_start,j=without_start], u[i]-u[j]+1 <= (N-1)*(1-x[i,j]))

    if op.start != op.stop
        @addConstraint(m, sum{x[op.stop,i],i=1:N} == 0)
    end

    status = solve(m)

    if status != :Optimal
        warn("Not solved to optimality:\n$op")
    end

    soln = OPSolution(getValue(x), getValue(u))

    if distance(op,build_path(op,soln)) > op.distance_limit
        warn("Path Length: $(distance(op,build_path(op,soln))), Limit: $(op.distance_limit)")
    end
    return soln
end

function gurobi_solve(op; output=0, initial::OPSolution=OPSolution())
    m = Model(solver=Gurobi.GurobiSolver(OutputFlag=output))
    N = length(op) # defined as the number of vertices in the problem.

    without_start = [1:op.start-1; op.start+1:N]
    without_stop = [1:op.stop-1; op.stop+1:N]
    without_both = intersect(without_start, without_stop)

    @defVar(m, x[1:N,1:N], Bin) #NxN array of binary variables - x[i,j] == 1 means j is visited just after i
    @defVar(m, 2 <= u[without_start] <= N, Int) # is u defined here, or elsewhere? u is degree?

    # For hot-starting solver?
    if !is(initial.x, nothing)
        setValue(x, initial.x)
        for keytuple in keys(u)
            key = keytuple[1]
            setValue(u[key], initial.u[key])
        end
    end

    # sum reward of visited nodes
    @setObjective(m, Max, sum{ sum{op.r[i]*x[i,j], j=1:N}, i=1:N })

    # limit one child per parent
    @addConstraint(m, sum{x[op.start,j], j=without_start} == 1)
    @addConstraint(m, sum{x[i,op.stop], i=without_stop} == 1)

    # problem
    @addConstraint(m, connectivity[k=without_both], sum{x[i,k], i=1:N} == sum{x[k,j], j=1:N})

    @addConstraint(m, once[k=1:N], sum{x[k,j], j=1:N} <= 1)
    @addConstraint(m, sum{ sum{op.distances[i,j]*x[i,j], j=1:N}, i=1:N } <= op.distance_limit)
    @addConstraint(m, nosubtour[i=without_start,j=without_start], u[i]-u[j]+1 <= (N-1)*(1-x[i,j]))

    if op.start != op.stop
        @addConstraint(m, sum{x[op.stop,i],i=1:N} == 0)
    end

    status = solve(m)

    if status != :Optimal
        warn("Not solved to optimality:\n$op")
    end

    soln = OPSolution(getValue(x), getValue(u))

    if distance(op,build_path(op,soln)) > op.distance_limit
        warn("Path Length: $(distance(op,build_path(op,soln))), Limit: $(op.distance_limit)")
    end
    return soln
end


# Test on a simple problem instance
function test_run()

    println("Creating problem instance")
    # Create a simple problem instance
    r = [0., 1., 1., 0., 1.99, 0.]
    start = 1
    stop  = 6
    # The graph looks like this:
    #   o - o 
    # o       o
    #   o - o 
    positions = Vector{Float64}[[0.,1.],[1.,2.], [1.+sqrt(2), 2.],[1.,0.], [1.+sqrt(2),0.], [0.,3.]]
    budget = 3*sqrt(2) + 0.1
    s2 = sqrt(2)
    distances = [0. s2 budget s2 budget budget;
                 s2 0. s2 budget budget budget;
                 budget s2 0. budget budget s2;
                 s2 budget budget 0. s2 budget;
                 budget budget budget s2 0. s2;
                 budget budget s2 budget s2 0.]
    
    # create OP instance
    op = SimpleOP(r, positions, budget, start, stop, distances);
   
    # Gurobi solver
    solver=GurobiExactSolver()
    println("Attempting to solve")
    # solve OP
    sol = solve_op(solver, op)
    println("Solution is $sol")

end
