using CuArrays, CUDAnative, CUDAdrv
using Test

# Kernel
function add!(a,b,c, index)
    c[index] = a[index] + b[index]
    return nothing
end

function main()

    # Initialise a, b and c
    a = CuArrays.CuArray(fill(0, 10))
    b = CuArrays.CuArray(fill(0, 10))
    c = CuArrays.CuArray(fill(0, 10))

    # Put values in a and b
    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    # Create two streams
    s1 = CuStream()
    s2 = CuStream()

    # Call add! asynchronously in two streams
    for i in 1:2:min(length(a), length(b), length(c))
        @cuda threads = 1 stream = s1 add!(a,b,c, i)
        @cuda threads = 1 stream = s2 add!(a,b,c, i+1)
    end

    # Copy arrays back to host (CPU)
    a=Array(a)
    b=Array(b)
    c=Array(c)

    # Check the addition worked
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end

end

main()
