using Test

function add!(a,b,c)
    local tid = 1
    while (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
        tid += 1
    end
end

function main()

    # Make three vectors
    a = Vector{Any}(fill(0, 10))
    b = Vector{Any}(fill(0, 10))
    c = Vector{Any}(fill(0, 10))

    # Fill a and b with values
    for i in 1:10
        a[i] = i
        b[i] = i * 2
    end

    # Fill c with values
    add!(a,b,c)

    # Do a sanity check
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end
end

main()
