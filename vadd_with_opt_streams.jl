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
buf_b2 = Mem.alloc(sizeof(Float32))

#Allocate memory for a and b on device stream 2
buf_a2 = Mem.alloc(sizeof(Float32))
buf_b2 = Mem.alloc(sizeof(Float32))

#Allocate memory for c on device
d_c1 = Mem.alloc(sizeof(Float32))
d_c2 = Mem.alloc(sizeof(Float32))

#Make streams
s1 = CuStream().handle
s2 = CuStream()

#Iterate over arrays in increments of 2
for i in 1:dims

    #Asynchronously copy a[i] and b[i] onto device
    Mem.upload!(buf_a1, a[i], async = true, stream = s1)
    Mem.upload!(buf_b1, b[i], async = true, stream = s1)

    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a1, buf_b1, d_c1,
    threads = 1, stream = s1)

    #Asynchronously copy c back to host
    Mem.download!(c[i], d_c, async = true, stream = s1)

end

#Check it worked
@test a+b ≈ c

#Destroy context
destroy!(ctx)
