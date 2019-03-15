using CuArrays, CUDAnative, CUDAdrv
using Test

function add!(a,b,c, index)
    c[index] = a[index] + b[index]
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

    s1 = CuStream()
    s2 = CuStream()

    for i in 1:2:min(length(a), length(b), length(c))

        @cuda threads = 1 stream = s1 add!(a,b,c, i)
        @cuda threads = 1 stream = s2 add!(a,b,c, i+1)
    end

    a=Array(a)
    b=Array(b)
    c=Array(c)

    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end

end

main()
