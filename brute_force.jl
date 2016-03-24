# brute force solve the problem using BFS. 
# This is VERY memory intensive, so don't try it on a large problem!

using DataStructures
using FixedSizeArrays
type Node
  index::UInt8 # The computer won't be able to compute a 16x16 solution, so this should be good enough.
  neighbors::Vector{UInt8}
end


# Enumerate the paths.
function BFS(N)
    if(N*N > 256)
        error("N is too large - will cause overflow and will not finish in reasonable time (Try N < 16)");
    end
    node_vals = ones(N*N)
    B = 100
    b_min = 1;

    # BFS algorithm initialization
    NodeList = Vector{Node}(N^2);
    for i = 1:N^2
        NodeList[i] = Node(UInt8(i), Vector{UInt8}())
    end

    # make neighbor lists
    for i1 = 1:N
        for i2 = 1:N
            n1 = sub2ind((N,N),i1,i2);
            if(i1 < N)
                nright = sub2ind((N,N), i1+1,i2);
                NodeList[n1].neighbors = [NodeList[n1].neighbors;nright]
                NodeList[nright].neighbors = [NodeList[nright].neighbors;n1]
            end
            if(i2 < N)
                nup = sub2ind((N,N), i1,i2+1);
                NodeList[n1].neighbors= [NodeList[n1].neighbors;nup];
                NodeList[nup].neighbors= [NodeList[nup].neighbors;n1];
            end
        end
    end

# Print neighborhoods
#    for i1 = 1:N
#        for i2 = 1:N
#            print("($i1,$i2) = { ");
#            for i in NodeList[sub2ind((N,N),i1,i2)].neighbors
#                ii1,ii2 = ind2sub((N,N),i);
#                print("($ii1,$ii2) ");
#            end
#            println("}");
#        end
#    end

    Q = Queue(Vector{UInt8});
    goal = sub2ind((N,N), N,N);
    enqueue!(Q, [UInt8(sub2ind((N,N), 1,1))])

    max_val = Float64(0);
    max_path = Vector{UInt8}();
    max_budget = Float64(0)
    status_tracker = 0;
    while(!isempty(Q))
        #Print status for sanity
        status_tracker+=1;
        if(mod(status_tracker,100000)==0)
           print(".");
        end
        if(mod(status_tracker,10000000)==0)
           println()
        end

        # Pop path from queue, check if complete
        current = dequeue!(Q)
        last_node = NodeList[current[end]];
        if(last_node.index == goal)
            # check optimality of path
            node_path = Vector{Node}(length(current))
            L = length(current);
            f = ones(L)
            for i = 1:L
            	node_path[i] = NodeList[current[i]] 
                f[i] = node_vals[current[i]]
            end

            # Optimize budget allocation:
            path = current;
            sortperm!(path, f, rev=true)
            num_active_constraints = UInt8(0);
            # Find number of active constraints:
            b_k = Float64(0);
            k = L+1
            while(b_k < b_min && k > 1)
                k-=1;
                gamma = norm(f[1:k]./f[k])^2
                b_k = (B- (L-k)*b_min)/gamma
            end
            num_active_constraints = k
            b_used = (B-(L-k)*b_min)/sum((f[1:k]./f[k]).^2); 

            # Compute budget allocations, using Lagrangian solution
            split = b_min*ones(Float64,L);
            for i = 1:num_active_constraints
                split[i] = b_k*f[i]^2/(f[num_active_constraints]^2)
            end
            val = (f'*sqrt(split))[1]
            if(val > max_val)
                max_val = val;
                max_path = node_path;
                max_budget = sum(split)
            end
        else
        # Expand path if not at goal yet 
            for n in last_node.neighbors
                if !(n in current)
                    L = length(current);
                    newpath = Vector{UInt8}(L+1);
                    for ii = 1:L
                        newpath[ii]=current[ii];
                    end
                    newpath[L+1] = n;
                    enqueue!(Q, newpath)
                end
            end
        end
    end 

    println("\n$status_tracker paths computed. Maximum objective value is $max_val, used $max_budget budget, and the path is ");

    endpt = max_path[end];
    max_path = max_path[1:end-1]
    for i in max_path
        i1,i2 = ind2sub((N,N), i.index);
        print("($i1,$i2) => ");
    end
    i1,i2 = ind2sub((N,N),endpt.index)
    println("($i1,$i1)");

#    return max_val, max_path
end

