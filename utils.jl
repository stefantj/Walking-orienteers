
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


# generates a simple grid with N^2 points in [0,1]
function generate_grid(N; startnode=[0.,0.], endnode=[0.,1.])
    positions = Vector{Vector{Float64}}()
    scaling=1.0

    push!(positions, Vector{Float64}(startnode))
    push!(positions, Vector{Float64}(endnode))
    # Generate positions via two superimposed grids
    for xpos = 1:N
        for ypos = 1:N
            xdelta = 0.
            pos = [float((xpos-1)/(N)*scaling), float((ypos-1)/(N)*scaling)]
            if( norm(startnode - pos) < float(N)^(-2) || norm(endnode-pos) < float(N)^(-2))
            else
                push!(positions, Vector{Float64}(pos))
            end
        end
    end



    # Compute distances
    # this is a stupid way to do this but you don't have to do it often so it doesn't matter too much.
    L = length(positions)
    distances = 1000000*ones(L,L); #large number - should really put something better here.
    dmin = 1/(N)
    i1 = 0
    for p1 in positions
        i1+=1
        i2 = 0
        for p2 in positions
            i2+=1
            # check if distance is small
            if(norm(p1-p2) <= dmin + 1e-5)
                distances[i1,i2] = norm(p1-p2);
                distances[i2,i1] = norm(p1-p2);
            end
        end
    end
    return positions, distances
end


# currently broken. Yay.
# decimate_pointset(pointset, distances, sn, en, k)
#  Returns every kth point from pointset, evenly spaced.
#  Returns corresponding start, end nodes as in original graph
function decimate_pointset(pointset, distances, sn, en, k,N)
    decimated_positions = Vector{Vector{Float64}}()

    scaling = 1.0 # should autodetect this.

    #now generate points skipping k in every direction

    # Generate positions via two superimposed grids
    for xpos = 1:k:(N+k-1)
        if(xpos >= N)
            xpos = N
        end
        for ypos = 1:k:(N+k-1)
            if(ypos >= N)
                ypos = N
            end
            xdelta = 0.
            pos = [float((xpos-1)/(N-1)*scaling), float((ypos-1)/(N-1)*scaling)]
            push!(decimated_positions, Vector{Float64}(pos))
        end
    end

    delta = 1/float(2*(N-1));
    for xpos = 1:k:N-2+k
        if(xpos>= N-1)
            xpos=N-1
        end
        for ypos = 1:k:N+k-2
            if(ypos>= N-1)
                ypos=N-1
            end
            pos = [float((xpos-1)/(N-1)*scaling+delta), float((ypos-1)/(N-1)*scaling+delta)]
#            push!(decimated_positions, Vector{Float64}(pos))
        end
    end

    return decimated_positions
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
function print_pointset(pointset)
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
    
    C_PXY = ones(2*N) # x,y values the same
    [C_PXY[pos] = float(pos-1)/(2*N-1) for pos = 1:(2*N)]
    C_P = zeros(2*N,2*N)
    for pp in opt_path
        iter+=1
        # For 3D line plot
        p_x[iter] = pointset[pp][2]
        p_y[iter] = pointset[pp][1]
        p_z[iter] = (opt_alloc[pp]/bud_step )

        # For contour plot
        i_y = Int(floor(p_x[iter]*2*(N-1))+1)
        i_x = Int(floor(p_y[iter]*2*(N-1))+1)
        C_P[i_x,i_y] = opt_alloc[pp]/bud_step

    end    

    #plot3D(p_x,p_y,p_z);
    #contour3D(C_XY,C_XY,C_Z./reward_scaling)
    fig = figure("pyplot_surfaceplot", figsize=(10,10));
    ax = fig[:add_subplot](1,2,1, projection = "3d")
    ax[:contour](C_XY, C_XY, C_Z, 10, cmap = ColorMap("coolwarm"))
    ax[:plot3D](p_x, p_y, p_z) #3D line
    ax[:plot3D](p_x, p_y, 1+0*p_z, color="black") #projection onto z-plane
#    ax[:plot3D](0*p_x, p_y, p_z, color="gray")  #projection onto x-plane
#    ax[:plot3D](p_x, 1+0*p_y, p_z, color="gray")#projection onto y-plane


   
    ax[:view_init](45,45)
    ax[:set_zlim]([-0.000001, maximum(p_z)+1]);
    zlabel("Number of visits");

    # update figure
    fig[:canvas][:draw]()
end


