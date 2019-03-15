using CUDAdrv
using Test

#'Turn on' device
dev = CuDevice(0)
ctx = CuContext(dev)

#Read in C code
md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

#Make data
dims = 100
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

#Allocate memory for a and b on device stream 1
buf_a1 = Mem.alloc(sizeof(Float32))
buf_b1 = Mem.alloc(sizeof(Float32))

#Allocate memory for a and b on device stream 2
buf_a2 = Mem.alloc(sizeof(Float32))
buf_b2 = Mem.alloc(sizeof(Float32))

#Allocate memory for c on device
d_c1 = Mem.alloc(sizeof(Float32))
d_c2 = Mem.alloc(sizeof(Float32))

#Make streams
s1 = CuStream()
s2 = CuStream()

#Iterate over arrays in increments of 2
for i in 1:2:dims

    #Asynchronously copy a[i] and a[i+1] onto device
    Mem.upload!(buf_a1, Ref(a, i), sizeof(Float32), s1, async = true)
    Mem.upload!(buf_a2, Ref(a, i+1), sizeof(Float32), s2, async = true)

    #Asynchronously copy b[i] and b[i+1] onto device
    Mem.upload!(buf_b1, Ref(b, i), sizeof(Float32), s1, async = true)
    Mem.upload!(buf_b2, Ref(b, i+1), sizeof(Float32), s2, async = true)

    #Call vadd to run on gpu
    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a1, buf_b1, d_c1,
    threads = 1, stream = s1)
    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a2, buf_b2, d_c2,
    threads = 1, stream = s2)

    #Asynchronously copy c[i] and c[i+1] back to host
    Mem.download!(Ref(c, i), d_c1, sizeof(Float32), s1, async = true)
    Mem.download!(Ref(c, i+1), d_c2, sizeof(Float32), s2, async = true)


end

#Check it worked
@test a+b ≈ c

#Destroy context
destroy!(ctx)
