using CUDAdrv

using Test

# 'Turn on' device
dev = CuDevice(0)
ctx = CuContext(dev)

# Read in C code
md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

# Make data
dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

# Allocate memory for a and b on device
buf_a = Mem.alloc(a)
buf_b = Mem.alloc(b)

# Asynchronously copy a and b onto device
Mem.upload!(buf_a, a, async = true)
Mem.upload!(buf_b, b, async = true)

# Allocate memory for c on device
d_c = Mem.alloc(c)

# Run kernel
len = prod(dims)
s = CuStream()
cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a, buf_b, d_c, threads = len, stream = s)

# Asynchronously copy c back to host
Mem.download!(c, d_c, async = true)

# Check it worked
@test a+b ≈ c

# Destroy context
destroy!(ctx)
