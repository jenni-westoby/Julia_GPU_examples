function add!(a,b,c)
    local tid = 1
    while (tid <= length(a))
        c[tid] = a[tid] + b[tid]
        tid += 1
    end
end

function main()

    a = Vector{Any}(fill(undef, 10))
    b = Vector{Any}(fill(undef, 10))
    c = Vector{Any}(fill(undef, 10))

    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    add!(a,b,c)

    for i in 1:length(a)
        println(a[i], " + ", b[i], " = ", c[i])
    end
end

main()
