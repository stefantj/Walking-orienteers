# This is a quick-and-dirty implementation of the walking orienteers algorithm.

include("OP_Solver.jl")
using OP_Solver
using UnicodePlots

# helper function to avoid some weird overflow-like symptoms
function sum_path_dict(path, times_dict)
    s = 0;
    [s+=times_dict[p] for p in path]
    return s
end


function waterfilling(path, rewards, budget, min_step)
    # efficient time allocation lookup
    times = Dict(zip(path, min_step*ones(path)));
    b_init = sum_path_dict(path, times)
    budget_left = budget;

    # Priority queue to keep track of best next step
    PQ = Collections.PriorityQueue()
    for p in path
        # increment of budget:
        PQ[p] = -rewards[p]*(sqrt(2*min_step)-sqrt(min_step))
    end

    # Greedy algorithm. Should probably improve?
    while (budget_left >= min_step)
        #largest node is at the top
        p_max = Collections.peek(PQ)[1] # index of biggest
        times[p_max] += min_step
        budget_left = budget_left - min_step
        PQ[p_max] = -rewards[p_max]*(sqrt(times[p_max]+min_step) - sqrt(times[p_max]))
    end
    if(sum_path_dict(path,times) > budget + b_init + 0.001)
        println("Error: used more budget than had allocated.")
        println(times)
    end
    # put remaining budget on best node
    if(budget_left > 0)
        p_max = Collections.peek(PQ)[1] # index of biggest
        times[p_max] += budget_left
        budget_left -= budget_left
    end

    # compute the reward
    obj_val = 0.
    for p in path
        obj_val += rewards[p]*sqrt(times[p])
    end

    return float(obj_val), times
end


# Generates N^2 + (N-1)^2 points in a [0,1]^dim square
# points are evenly spaced and only connected to their nearest neighbors
function generate_lattice(N; dim=2, scaling=1.0)
    positions = Vector{Vector{Float64}}()
    
    # Generate positions
    delta = 0.;

    # Two superimposed grids
    for xpos = 1:N
        for ypos = 1:N
            xdelta = 0.
            pos = [float((xpos-1)/(N-1)*scaling+delta), float((ypos-1)/(N-1)*scaling+delta)]
            push!(positions, Vector{Float64}(pos))
        end
    end

    delta = 1/float(2*(N-1));
    for xpos = 1:N-1
        for ypos = 1:N-1 
            pos = [float((xpos-1)/(N-1)*scaling+delta), float((ypos-1)/(N-1)*scaling+delta)]
            push!(positions, Vector{Float64}(pos))
        end
    end


# this is kind of a stupid way to do this but you don't have to do it often so it doesn't matter too much.
    L = length(positions)
    distances = 1000000*ones(L,L); #large number - should really put something better here.
    dmin = 1/(sqrt(2)*(N-1))
    i1 = 0
    for p1 in positions
        i1+=1
        i2 = 0
        for p2 in positions
            i2+=1
            # check if distance is small
            if(norm(p1-p2) <= dmin + 1e-5)
                distances[i1,i2] = norm(p1-p2);
            end 
        end
    end
    return positions, distances
end

function test()

    # Form lattice
    N = 20 # there are N^2 + (N-1)^2 points in the lattice
    pointset, distances = generate_lattice(N)

    println("Solving a problem with ", N^2+(N-1)^2, " nodes.")

    # Budget values
    B_MAX = 3;
    B_MIN = sqrt(2)+0.0001;
    B_step = 1/(sqrt(2)*(N-1))
    
    # Reward function
    r = ones(length(pointset))
    i = 0;
    for pt in pointset
        i+=1
       r[i] = exp(-10*norm(pt-[0.55,0.75])^2)
    end
    reward_vals = Dict(zip(pointset, r)) # for easy lookup

    # Form problem instance
    op = OP_Solver.SimpleOP(r, pointset ,B_MAX, 1, 400, distances);
    solver=OP_Solver.GurobiExactSolver()


println("\t B_path \t L_path \t total \t opt_val")

    # search over budget values
    max_val = -1
    max_path = [];
    max_alloc = Dict()
    for budget = B_MIN:B_step:B_MAX
        # solve OP for base path
        op.distance_limit = budget
        path = OP_Solver.solve_op(solver, op)
        used_budget = (length(path)-1)/(sqrt(2)*(N-1))
        bval = B_MAX - used_budget;

        # use water-filling to use the rest of the budget
        obj_val, budget_alloc = waterfilling(path, r, bval, B_step)        
        if(obj_val == -1) # means we've hit upper limit
            break
        end

        println(used_budget, "\t", sum_path_dict(path, budget_alloc) - used_budget, "\t", sum_path_dict(path, budget_alloc), "\t", obj_val)

        # Error checking
        if(sum_path_dict(path,budget_alloc) < 0)
            println("Error: allocated negative budget (", sum(budget_alloc.vals))
            println(budget_alloc)
        end

        # plot for debug
    xvec = zeros(length(path))
    yvec = zeros(length(path))
    i = 1
    for pp in path
        #println(pointset[pp][1], "\t", pointset[pp][2], "\t", max_alloc[pp])
        xvec[i] = pointset[pp][1]
        yvec[i] = pointset[pp][2]
        i+=1
    end
    print(scatterplot(xvec, yvec, xlim=[0,1], ylim=[0,1]))

        # Track if maximum
        if(obj_val > max_val)
            max_val = obj_val
            max_path = path
            max_alloc = budget_alloc
        end
    end

    println()
    println("Solution is")
    xvec = zeros(length(max_path))
    yvec = zeros(length(max_path))
    i = 1
    for pp in max_path
        println(pointset[pp][1], "\t", pointset[pp][2], "\t", max_alloc[pp])
        xvec[i] = pointset[pp][1]
        yvec[i] = pointset[pp][2]
        i+=1
    end
    print(scatterplot(xvec, yvec, xlim=[0,1], ylim=[0,1]))

    
end

