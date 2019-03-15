function add!(a,b,c)
    local tid = 1
    while (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
        tid += 1
    end
end

function main()

    # make three vectors
    a = Vector{Any}(fill(undef, 10))
    b = Vector{Any}(fill(undef, 10))
    c = Vector{Any}(fill(undef, 10))

    # Fill a and b with values
    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    # Fill c with values
    add!(a,b,c)

    # Do a sanity check
    for i in 1:length(a)
        println(a[i], " + ", b[i], " = ", c[i])
    end
end

main()
