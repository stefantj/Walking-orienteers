# Implementation of walking orienteers algorithm. 


include("OP_Solver.jl")
include("../Julia\ FMT/Libraries/GPR/GPR.jl")

using GPR
using OP_Solver
#using UnicodePlots
using PyPlot


#### Helper functions ####

# sum_path_dict(path, times_dict)
#  helper function to sum over a dict without directly accessing keys 
#  (avoids some weird overflow-like symptoms)
function sum_path_dict(path, times_dict)
    s = 0;
    [s+=times_dict[p] for p in path]
    return s
end

# generate_lattice(N)
#  Generates N^2 + (N-1)^2 points in a [0,1]^dim square
#  points are evenly spaced and only connected to their nearest neighbors
function generate_lattice(N; dim=2, scaling=1.0)
    positions = Vector{Vector{Float64}}()
    
    # Generate positions via two superimposed grids
    for xpos = 1:N
        for ypos = 1:N
            xdelta = 0.
            pos = [float((xpos-1)/(N-1)*scaling), float((ypos-1)/(N-1)*scaling)]
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

    # Compute distances
    # this is a stupid way to do this but you don't have to do it often so it doesn't matter too much.
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

# generate_Gaussian_reward(pointset)
#  samples a Gaussian process using GPR library
#  Can be generalized to include prior data by using
#  an existing GPR.GaussianProcessEstimate object.
function generate_Gaussian_reward(pointset)
    # Generate model
    noisevar = 0.01
    bandwidth = 0.1
    k_SE = GPR.SquaredExponential(bandwidth)
    model = GPR.GaussianProcess(noisevar, k_SE)
    estimator = GPR.GaussianProcessEstimate(model, 2)
    r = vec(GPR.sample_n(estimator, pointset))
    reward_vals = Dict(zip(pointset, r)) # for easy lookup
    return r, reward_vals
end

# generate_simple_reward(pointset)
#  generates rewards using a decaying exponential function. 
function generate_simple_reward(pointset)
    r = ones(length(pointset))
    i = 0;
    for pt in pointset
        i+=1
       r[i] = 1#*exp(-10*norm(pt-[0.55,0.75])^2)
    end
    r[53] = 10;

    reward_vals = Dict(zip(pointset, r)) # for easy lookup
    return r, reward_vals
end

#### 

#### Printing functions ####

function print_path(pointset::Vector{Vector{Float64}}, path::Array{Float64,1}, budget_alloc::Array{Float64,1})
    xvec = zeros(length(path))
    yvec = zeros(length(path))
    i = 1
    for pp in path
        println(pointset[pp][1], "\t", pointset[pp][2], "\t", budget_alloc[pp])
        xvec[i] = pointset[pp][1]
        yvec[i] = pointset[pp][2]
        i+=1
    end
end


function print_rewards(pointset, reward_vals)
   # print as (xyz) tuples
   println("reward_surface=[") 
   [println(pp[1],"\t",pp[2],"\t",reward_vals[pp]) for pp in pointset]
   println("]")
end

# deprecated
function print_pointset()
    N = 20 # there are N^2 + (N-1)^2 points in the lattice
    pointset, distances = generate_lattice(N)
    println("Pointset is")
    #print points for debug
    for pp in pointset
      println(pp[1],"\t",pp[2]);
    end
end

# assumes UnicodePlots loaded
function plot_path_alloc(path, budget_alloc)

   x = collect(1:length(path))
   y = zeros(length(path));

   iter = 0;
   for pp in path
      iter+=1
      y[iter] = budget_alloc[pp]
   end
#   println(scatterplot(x,y))
end

# assumes PyPlot loaded
function plot_solution(pointset, opt_path, opt_alloc, reward_vals, N)
    # generate reward surface contour
    # since a contour plot, use only the sparser N^2 grid.
    C_XY = ones(N) # x,y values the same
    [C_XY[pos] = float(pos-1)/(N-1) for pos = 1:N]
    C_Z = zeros(N,N)
    [[C_Z[x,y] = reward_vals[[C_XY[x],C_XY[y]]] for x=1:N] for y=1:N]

    # generate path data in plottable form
    p_x = zeros(length(opt_path))
    p_y = zeros(length(opt_path))
    p_z = zeros(length(opt_path))
    bud_step = 1/(sqrt(2)*(N-1))
    iter= 0;
    for pp in opt_path
        iter+=1
        p_x[iter] = pointset[pp][2]
        p_y[iter] = pointset[pp][1]
        p_z[iter] = (opt_alloc[pp]/bud_step ) #reward_vals[pointset[pp]]
    end

    #plot3D(p_x,p_y,p_z);
    #contour3D(C_XY,C_XY,C_Z./reward_scaling)
    fig = figure("pyplot_surfaceplot", figsize=(10,10));
    ax = fig[:add_subplot](1,1,1, projection = "3d")
    println(ax)
    ax[:plot3D](p_x, p_y, p_z)#, cmap=ColorMap("gray"))
    ax[:contour3D](C_XY, C_XY, C_Z, 10, zdir="z", offset=0, cmap = ColorMap("coolwarm"))
    zlabel("Number of visits");
    ax[:set_zlim]([-0.000001, maximum(p_z)+1]);

end


####

# waterfilling(path, rewards, budget, min_step)
#  Allocates remaining $budget in increments of $min_step 
#  along $path to maximize sum of $rewards, with sqrt
#  diminishing returns.
#  Solved using greedy algorithm.
#
function waterfilling(path, rewards, budget, min_step)
    # efficient time allocation lookup
    times = Dict(zip(path, min_step*ones(path)));
    b_init = sum_path_dict(path, times)
    budget_left = budget;
    increment = min_step/10;

    # Priority queue to keep track of best next step
    PQ = Collections.PriorityQueue()
    for p in path
        # increment of budget:
        PQ[p] = -rewards[p]*(sqrt(min_step+increment)-sqrt(min_step))
    end

    # Greedy algorithm. Should probably improve?
    while (budget_left >= increment)
        #largest node is at the top
        p_max = Collections.peek(PQ)[1] # index of biggest
        times[p_max] += increment
        budget_left = budget_left - increment
        PQ[p_max] = -rewards[p_max]*(sqrt(times[p_max]+increment) - sqrt(times[p_max]))
    end
#    if(sum_path_dict(path,times) > budget + b_init + 0.001)
#        println("Error: used more budget than had allocated.")
#        println(times)
#    end
    # put any remaining budget on best node
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



# Test function. 
# example implementation with debugging checks and outputs.
function test()
    println("starting test")
    # Form lattice
    N = 10 # there are N^2 + (N-1)^2 points in the lattice
    pointset, distances = generate_lattice(N)
    start_node = 1; # (0,0)
    end_node = N; # (1,0)

    println("Solving a problem with ", N^2+(N-1)^2, " nodes.")

    # Budget values
    B_MAX = 10;
    B_MIN = 1+0.0001;
    B_step = 1/(sqrt(2)*(N-1))

    # Reward function
    r, reward_vals = generate_Gaussian_reward(pointset)
#    r, reward_vals = generate_simple_reward(pointset)

    # Form problem instance
    op = OP_Solver.SimpleOP(r-minimum(r)+0.001, pointset ,B_MAX, start_node, end_node, distances);
    solver=OP_Solver.GurobiExactSolver()

    println("Solution log:\n \t B_path \t\t L_path \t\t total \t\t opt_val")

    # search over budget values
    max_val = -1
    max_path = [];
    max_alloc = Dict()
    budget = B_MIN;
    B_STEP = (B_MAX-B_MIN)/10;
    while(budget < B_MAX)
        budget += B_STEP; 
        if(budget > B_MAX)
            budget = B_MAX
        end
        # solve OP for base path
        op.distance_limit = budget
        path = OP_Solver.solve_op(solver, op)
        used_budget = (length(path))/(sqrt(2)*(N-1))
        remaining_budget = B_MAX - used_budget;

        println("Filling in remaining budget...")
        # use water-filling to use the rest of the budget
        obj_val, budget_alloc = waterfilling(path, r, remaining_budget, B_step)        
        if(obj_val == -1) # means we've hit upper limit
            break
        end

        println(used_budget, "\t", sum_path_dict(path, budget_alloc) - used_budget, "\t", sum_path_dict(path, budget_alloc), "\t", obj_val)

        # Error checking
        if(sum_path_dict(path,budget_alloc) < 0)
            println("Error: allocated negative budget (", sum(budget_alloc.vals))
            println(budget_alloc)
        end

        #println("Allocation profile (", length(path), length(budget_alloc.vals),")")
        #print_path_alloc(path, budget_alloc)

        #println("path$budget_index=[")
    	#print_path(pointset, path, alloc )
	#println("]");

        # Track if maximum
        if(obj_val > max_val)
            max_val = obj_val
            max_path = path
            max_alloc = budget_alloc
        end
    end

    println()
#    print_path(pointset, max_path, max_alloc )
#    println("Reward_surface=[")
#    [println(pp[1],"\t",pp[2],"\t",reward_vals[pp]) for pp in pointset]
#    println("]")

    plot_solution(pointset, max_path, max_alloc, reward_vals, N)
    return pointset, max_path, max_alloc, reward_vals, N
end


