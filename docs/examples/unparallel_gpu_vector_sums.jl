using CuArrays, CUDAnative, CUDAdrv

function add!(a,b,c)
    local tid = 1
    while (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
        tid += 1
    end
    return nothing
end

function main()

    a = CuArrays.CuArray(fill(0, 10))
    b = CuArrays.CuArray(fill(0, 10))
    c = CuArrays.CuArray(fill(0, 10))

    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    # IMPORTANT NOTE TO SELF
    # you can pass tuples to represent a grid to blocks, just like in CUDA C <3
    @cuda blocks=10 add!(a,b,c)

    a=Array(a)
    b=Array(b)
    c=Array(c)

    for i in 1:length(a)
        println(a[i], " + ", b[i], " = ", c[i])
    end
end

main()
