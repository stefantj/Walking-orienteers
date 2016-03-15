# brute force solve the problem using BFS. 
# This is VERY memory intensive, so don't try it on a large problem!

using DataStructures
using FixedSizeArrays

type Node
  index::UInt8 # The computer won't be able to compute a 16x16 solution, so this should be good enough.
  neighbors::Vector{UInt8}
end
    N = 6;

function print_path(list)
    last = list[end];
    list = list[1:end-1];
    for node in list
        i1,i2 = ind2sub((N,N), node.index);
        print("($i1,$i2) => ")
    end
    i1,i2 = ind2sub((N,N), last.index);
    println("($i1,$i2)");
end

function allocate_budget(node_list)
    # Extract node identities
    L = length(node_list)
    path = Vector{UInt8}(L)
    iter =0;
    for n in node_list
        iter+=1;
        path[iter] = UInt8(n.index);
    end


    # In future will need to input: f, B
    #function values
    f = ones(L) # in reality, this would be f(path)
    B = 100
    b_min = 1;
    # find out how many constraints are active
    path_unsorted = path;
    sortperm!(path, f, rev=true)
    split = b_min*ones(L);
    num_active_constraints = 1;
    beta = 1;
    b_k = b_min;
    while true
        gamma = 0;
        [gamma += (f[num_active_constraints]/f[i])^2 for i = 1:num_active_constraints]
        beta = (B-b_min*(L-num_active_constraints))/gamma;
        (beta < b_min || num_active_constraints == L)&&break
        b_k = beta
        num_active_constraints += 1;
    end
#    num_active_constraints -= 1

    for i = 1:num_active_constraints
        split[i] = b_k*f[i]^2/(f[num_active_constraints]^2)
    end
    
#    println("$num_active_constraints constraints with $b_k to all");
    reward = 0;
    [reward += f[i]^2*sqrt(b_k)/f[num_active_constraints] for i = 1:num_active_constraints]
    [reward += f[i]*sqrt(b_min) for i = num_active_constraints+1:L]
    return reward, path_unsorted, split
end

# Enumerate the paths.
function BFS()
    if(N*N > 256)
        error("N is too large - will cause overflow and will not finish in reasonable time (Try N < 16)");
    end

    # BFS algorithm initialization
    NodeList = Vector{Node}(N^2);
    [NodeList[i] = Node(UInt8(i), Vector{UInt8}()) for i = 1:N^2]

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

    Q = Queue(Vector{Node});
    goal = sub2ind((N,N), N,N);
    enqueue!(Q, [NodeList[sub2ind((N,N), 1,1)]])

    max_val = 0;
    max_path = [];

    status_tracker = 0;
    while(!isempty(Q))
        status_tracker+=1;
        if(mod(status_tracker,10000)==0)
           print(".");
        end
        if(mod(status_tracker,1000000)==0)
           println()
        end

        current = dequeue!(Q)
        last_node = current[end];
        if(current[end].index == goal)
            #check optimality of path
            val, path, split = allocate_budget(current)
            if(val > max_val)
                max_val = val;
                max_path = current;
                println("optval: $max_val, length ", length(max_path), " Budget ", sum(split))
            end
        end
 
        for n in last_node.neighbors
            if !(NodeList[n] in current)
                newpath = [current; NodeList[n]];
                enqueue!(Q, newpath)
            end
        end
    end 

    println("$status_tracker paths computed. Maximum objective value is $max_val, and the path is ");

    endpt = max_path[end];
    max_path = max_path[1:end-1]
    for i in max_path
        i1,i2 = ind2sub((N,N), i.index);
        print("($i1,$i2) => ");
    end
    i1,i2 = ind2sub((N,N),endpt.index)
    println("($i1,$i1)");

end

