# Implementation of walking orienteers algorithm. 

include("OP_Solver.jl")
include("utils.jl")
include("../Julia\ FMT/Libraries/GPR/GPR.jl")

using GPR
using OP_Solver
using PyCall
using PyPlot

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
    increment = min_step/1000;

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

function sample_reward_function(x::Vector{Float64})
    return sample_reward_function(x[1],x[2]);
end
# this is much easier than trying to make a stupid matching grid...
function sample_reward_function(xpt,ypt)
 centers = [0.5  0.5   1 
 		    0.702462  0.617413    1
		    0.276299  0.0144      1 
 		    0.612661  0.9574     -1
 		    0.855629  0.427364    1
            0.860654  0.0238044  -1
            0.401519  0.660306   -1
            0.280782  0.533708    1
            0.186563  0.396171    1
            0.404439  0.676824   -1
            0.700535  0.834689    1 
            0.367357  0.143921   -1
            0.339283  0.168406   -1 
            0.071649  0.688787    1
            0.902341  0.39723    -1]
    reward_val = 0;
    for k=1:15
        reward_val +=centers[k,3]*exp(- ((xpt-centers[k,1])^2 + (ypt-centers[k,2])^2)/0.03)
    end
    return reward_val;
end

function discretization_test()
    start_node = 1;
    end_node = 2;
    N=20; 
    pointset, distances = generate_grid(N)
    # reshape for convenience
    x_vals = zeros(N^2+2)
    y_vals = zeros(N^2+2)
    iter = 0;
    for pp in pointset
        iter+=1;
        x_vals[iter] = pp[1]
        y_vals[iter] = pp[2]
    end

    R_img = zeros(100,100);
    for xindex = 1:100
        for yindex = 1:100
            R_img[yindex,xindex] = sample_reward_function( (xindex-1)/100, 1 - (yindex-1)/100  )
        end
    end
    #imshow(R_img, extent=[0,1,0,1])
    
    budget = 30;

    for n = 5:20
        #Sample points
        decimated_points, decimated_distances = generate_lattice(n)

        #Sample reward function
        r_decimated = zeros(length(decimated_points))
        for i=1:length(decimated_points)
            r_decimated[i] = sample_reward_function(decimated_points[i])
        end

        # Compute path for given discretization

       println("Traveling from ", decimated_points[1], " to ", decimated_points[n])
       # Form problem instance
       op = OP_Solver.SimpleOP(r_decimated-minimum(r_decimated)+0.001, decimated_points,1.01, 1,n, decimated_distances);
       solver=OP_Solver.GurobiExactSolver()

       # compute path
       path = OP_Solver.solve_op(solver, op)

    	p_x = zeros(length(path))
    	p_y = p_x
    	iter= 0;
    	for pp in path
            iter+=1
            # For 3D line plot
            p_x[iter] = decimated_points[pp][2]
            p_y[iter] = decimated_points[pp][1]
    	end


        # points for plotting
        x_dec = zeros(length(decimated_points))
        y_dec = zeros(length(decimated_points))

        iter = 0;
        for pp in decimated_points
            iter+=1;
            x_dec[iter] = pp[2]; y_dec[iter]=pp[1];
        end
  
        # plot results and save frame
        fig = figure(2);
        cla()
        #axis([0,1,0,1])
#        scatter(x_vals, y_vals, color="black", hold=true)
        scatter(x_dec, y_dec, color="red")
        imshow(R_img, extent=[0,1,0,1])
        plot(p_x,p_y, color="black")
        numstring = string(n);
        if(length(numstring) == 1)
            numstring = string("00",numstring)
        elseif(length(numstring) == 2)
            numstring = string("0",numstring)
        end
        fname = string("img", numstring, ".png")
        println(fname)
        savefig(fname)
    end
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
        else
            println("Finished search?")
            break;
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


function test_exact()
    println("starting test")
    # Form lattice
    N = 10 # there are N^2 + (N-1)^2 points in the lattice
    pointset, distances = generate_lattice(N)
    start_node = 1; # (0,0)
    end_node = N; # (1,0)

    println("Solving a problem with ", N^2+(N-1)^2, " nodes.")

    # Budget values
    B_MAX = 10;

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
    op.distance_limit = B_MAX
    path = OP_Solver.solve_exact_op(solver, op, 1/sqrt(N))
    println("I guess it worked?")


end
