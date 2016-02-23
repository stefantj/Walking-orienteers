module OP_Solver
    import Base: length

    export SimpleOP,
        GurobiExactSolver
    export solve_op,
        reward,
        distance,
        build_path,
        gurobi_solve,
        test_run

    
    type SimpleOP
        r::Vector{Float64}                 # reward vector
        positions::Vector{Vector{Float64}} # position vector
        distance_limit::Float64            # Budget constraint
        start::Int                         # start index
        stop::Int                          # stop index
        distances::Matrix{Float64}         # inter-vertex distance
    end

    reward(op::SimpleOP, path::Vector{Int}) = sum([op.r[i] for i in path])
    Base.length(op::SimpleOP) = length(op.r)
    distance(op, path::Vector{Int}) = sum([op.distances[path[i],path[i+1]] for i in 1:length(path)-1])


    include("solutions.jl")


end # module OP_solver
