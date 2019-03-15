using CuArrays, CUDAnative, CUDAdrv

# This is the kernel. The plus and minus ones relative to the C version are due
# to the fact that ONE INDEXING SUCKS
function dot(a,b,c, N, threadsPerBlock, blocksPerGrid)

    # Set up shared memory cache for this current block. Has to be dynamic to be
    # able to take command line args - this will not work with
    # @cuStaticSharedMem
    cache = @cuDynamicSharedMem(Int64, threadsPerBlock)

    # Initialise some variables. Minus ones are to get the indexing consistent
    # with the C version
    tid = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    cacheIndex = threadIdx().x - 1
    temp::Int64 = 0

    # iterate over vector to do dot product in parallel way
    while tid < N
        temp += a[tid + 1] * b[tid + 1]
        tid += blockDim().x * gridDim().x
    end

    # set cache values
    cache[cacheIndex + 1] = temp

    # synchronise threads
    sync_threads()

    # In the step below, we add up all of the values stored in the cache
    i::Int = blockDim().x/2
    while i!=0
        if cacheIndex < i
            cache[cacheIndex + 1] += cache[cacheIndex + i + 1]
        end
        sync_threads()
        i/=2
    end

    # cache[1] now contains the sum of vector dot product calculations done in
    # this block, so we write it to c
    if cacheIndex == 0
        c[blockIdx().x] = cache[1]
    end

    return nothing
end

# Self explanatory
function sum_squares(x)
    return (x * (x + 1) * (2 * x + 1) / 6)
end


function main()

    # Can't type global variables in Julia, which messes things up.
    # So instead we locally define these and pass them explicitly as arguments to
    # the kernel
    N::Int64 = 33 * 1024
    threadsPerBlock::Int64 = 256
    blocksPerGrid::Int64 = min(32, (N + threadsPerBlock - 1) / threadsPerBlock)

    # Create a,b and c
    a = CuArrays.CuArray(fill(0, N))
    b = CuArrays.CuArray(fill(0, N))
    c = CuArrays.CuArray(fill(0, blocksPerGrid))

    # Fill and b
    for i in 1:N
        a[i] = i
        b[i] = 2*i
    end

    # Execute the kernel. Note the shmem argument - this is necessary to allocate
    # space for the cache we allocate on the gpu with @cuDynamicSharedMem
    @cuda blocks = blocksPerGrid threads = threadsPerBlock shmem =
    (threadsPerBlock * sizeof(Int64)) dot(a,b,c, N, threadsPerBlock, blocksPerGrid)

    # Copy c back from the gpu (device) to the host
    c = Array(c)

    local result = 0

    # Sum the values in c
    for i in 1:blocksPerGrid
        result += c[i]
    end

    # Check whether output is correct
    println("Does GPU value ", result, " = ", 2 * sum_squares(N - 1))
end

main()
