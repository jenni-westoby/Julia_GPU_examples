using CuArrays, CUDAnative, CUDAdrv, Test

function add!(a,b,c)
    tid = threadIdx().x
    if (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
    end
    return nothing
end

function main()

    # Make three CuArrays
    a = CuArrays.CuArray(fill(0, 10))
    b = CuArrays.CuArray(fill(0, 10))
    c = CuArrays.CuArray(fill(0, 10))

    # Fill a and b with values
    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    # Execute the kernel
    @cuda threads=10 add!(a,b,c)

    # Copy a,b and c back from the device to the host
    a = Array(a)
    b = Array(b)
    c = Array(c)

    # Do a sanity check
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end
end

main()
