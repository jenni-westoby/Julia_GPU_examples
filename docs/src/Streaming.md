# Streaming

In CUDA, a stream is a sequence of operations executed in order on a device. We can use multiple streams to execute multiple sequences of operations sequentially. A common reason for using multiple streams in GPU programming is to 'hide' the time taken for data transfer. Often copying data between the host and the device is one of the slowest steps in a GPU program. By writing your program with streams, you can split your data into chunks and have the GPU analysing a chunk in one stream whilst simultaneously copying a chunk of data to the GPU in another stream.

It should be noted that Julia's support for streaming in GPU programming is still rudimentary. As you will see, it is easy to stream kernel execution (analysis), but the ideal of streaming both data transfer and analysis is more challenging. Streaming both data transfer and analysis will require us to write much lower level code than is usually seen in Julia. Let's start with the simpler task of streaming our analysis.

# Streaming our Analysis

In this example, we will return to our favourite problem of vector addition. A script which will carry out vector addition in two streams is shown below:

```
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
```

So what have we changed to introduce streaming to the classic vector addition example? Firstly, the kernel is slightly different:

```
# Kernel
function add!(a,b,c, index)
    c[index] = a[index] + b[index]
    return nothing
end
```

We now pass the index into the vectors as an argument, rather than relying on the thread or block index.

As previously, ```main()``` begins by creating ```a```, ```b``` and ```c``` and putting values in ```a``` and ```b```. The next change is halfway through ```main()```, where we create two streams:

```
# Create two streams
s1 = CuStream()
s2 = CuStream()
```

This is fairly self explanatory - ```CuStream()``` is a function from CUDAnative which creates a stream. Next, we use the two streams to execute the kernel simultaneously in multiple streams:

```
# Call add! asynchronously in two streams
for i in 1:2:min(length(a), length(b), length(c))
    @cuda threads = 1 stream = s1 add!(a,b,c, i)
    @cuda threads = 1 stream = s2 add!(a,b,c, i + 1)
end
```

Let's walk through what this for loop is doing. This for loop iterates over the length of the vectors in steps of size 2. In each iteration, we execute the kernel once in the first stream (```s1```) using ```i``` as an index for vector addition, and once in the second stream (```s2```) using ```i + 1``` as the index for vector addition. Thus, in each interation of the for loop the values of ```c[i]``` and ```c[i + 1]``` are calculated in seperate streams. ```main()``` then finishes as before by copying ```a```, ```b``` and ```c``` back to the host and checking that the results of the calculation are correct.

Hopefully you agree that executing the kernel across two streams was extremely easy in this example. However, it was also fairly pointless. We basically wrote some extra code to execute add! on two elements of c at a time, whereas our previous example without streaming calculated every element of c in parallel using threads.

Streaming can really come in to its own when used to stagger data transfer and analysis between streams. This is possible in Julia for GPU applications, but comes with a health warning...

# Health Warning: Low Level Code Alert

To stagger data transfer and analysis between streams, we will not be able to use CUDAnative or CUarrays. This is because ```CuArrays.CuArray()```, the function we have used to copy data from host to device up to now, executes synchronously. This means that the function does not return until the data transfer is complete. If we try to stream data transfer with a synchronous data transfer function, we will not acheive any speed up because no other streams can receive and start executing any further instructions until the current stream's data transfer is complete and the function has returned.

To acheive speed-up by streaming data transfer, the process of data transfer must be asynchronous. If our data transfer function is asynchronous, it can return before finishing data transfer and we can submit other instructions to other streams whilst data transfer is still ongoing in the first stream. This can be a powerful way to speed up programs. It can also be useful if our data is too large to load on to the GPU all at once, as now we can upload and analyse our data in chunks whilst minimising the time the GPU spends sitting idle waiting for data to finish copying.

Unlike CuArrays and CUDAnative, CUDAdrv provides support for asynchronous data transfer. The code in the next example is very analogous to CUDA C or C++ code and the example below does in fact include a (very short) CUDA C script. In practice, this means we will have to think about memory management more than previously and our code will be less pretty. If you have never written code in a lower level language like C, you may struggle to follow the next section. If you are struggling, do not panic and just move on to the next section. In practice, you can often speed up your code a lot by porting to GPU without using streams.

# Streaming Data Transfer and Analysis

Let's rewrite our vector addition example so that it uses streaming for both data transfer and analysis.

```
using CUDAdrv, Test

# 'Turn on' device
dev = CuDevice(0)
ctx = CuContext(dev)

# Read in C code
md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")

# Make data
dims = 100
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

# Allocate memory for a and b on device stream 1
buf_a1 = Mem.alloc(sizeof(Float32))
buf_b1 = Mem.alloc(sizeof(Float32))

# Allocate memory for a and b on device stream 2
buf_a2 = Mem.alloc(sizeof(Float32))
buf_b2 = Mem.alloc(sizeof(Float32))

# Allocate memory for c on device
d_c1 = Mem.alloc(sizeof(Float32))
d_c2 = Mem.alloc(sizeof(Float32))

# Make streams
s1 = CuStream()
s2 = CuStream()

# Iterate over arrays in increments of 2
for i in 1:2:dims

    # Asynchronously copy a[i] and a[i+1] onto device
    Mem.upload!(buf_a1, Ref(a, i), sizeof(Float32), s1, async = true)
    Mem.upload!(buf_a2, Ref(a, i+1), sizeof(Float32), s2, async = true)

    # Asynchronously copy b[i] and b[i+1] onto device
    Mem.upload!(buf_b1, Ref(b, i), sizeof(Float32), s1, async = true)
    Mem.upload!(buf_b2, Ref(b, i+1), sizeof(Float32), s2, async = true)

    # Call vadd to run on gpu
    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a1, buf_b1, d_c1,
    threads = 1, stream = s1)
    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a2, buf_b2, d_c2,
    threads = 1, stream = s2)

    # Asynchronously copy c[i] and c[i+1] back to host
    Mem.download!(Ref(c, i), d_c1, sizeof(Float32), s1, async = true)
    Mem.download!(Ref(c, i+1), d_c2, sizeof(Float32), s2, async = true)


end

# Check it worked
@test a+b ≈ c

# Destroy context
destroy!(ctx)
```

This is probably looking pretty alien at this point - don't worry, we will work through it slowly. Starting from the beginning:

```
using CUDAdrv, Test

# 'Turn on' device
dev = CuDevice(0)
ctx = CuContext(dev)
```

The script starts as usual by loading some Julia packages. Then there are two lines we haven't seen before calling ```CuDevice``` and ```CuContext```. These two lines essentially work together to create a context on the GPU. A GPU context can be thought of as analogous to a CPU process, so by running these two lines we are creating an address space and allocated resources on the GPU where our kernel can run and we can copy data. Once we have finished working on the GPU, we will need to ```destroy``` our context so the system can clean up the resources allocated there. Both creating the context and destroying it are taken care of behind the scenes by CUDAnative, which is why we have been able to ignore these functions until now.

The lines of code that follow context creation will either delight or alarm you, depending on your relationship with C:

```
# Read in C code
md = CuModuleFile(joinpath(@__DIR__, "vadd.ptx"))
vadd = CuFunction(md, "kernel_vadd")
```

So what is going on here? Well, first ```CuModuleFile``` reads in a file called vadd.ptx. In the next line, we use ```CuFunction``` to make a handle to a function called ```kernel_vadd``` defined in vadd.ptx, then assign that function handle to ```vadd```. The outcome of these two lines is that a function called ```kernel_vadd``` defined in vadd.ptx becomes callable in Julia, and can be called using ```vadd()```.

So what do the contents of vadd.ptx look like?

```
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-23083092
// Cuda compilation tools, release 9.1, V9.1.85
// Based on LLVM 3.4svn
//

.version 6.1
.target sm_30
.address_size 64

	// .globl	kernel_vadd

.visible .entry kernel_vadd(
	.param .u64 kernel_vadd_param_0,
	.param .u64 kernel_vadd_param_1,
	.param .u64 kernel_vadd_param_2
)
{
	.reg .f32 	%f<4>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd1, [kernel_vadd_param_0];
	ld.param.u64 	%rd2, [kernel_vadd_param_1];
	ld.param.u64 	%rd3, [kernel_vadd_param_2];
	cvta.to.global.u64 	%rd4, %rd3;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd1;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	mul.wide.s32 	%rd7, %r4, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.f32 	%f1, [%rd8];
	add.s64 	%rd9, %rd5, %rd7;
	ld.global.f32 	%f2, [%rd9];
	add.f32 	%f3, %f1, %f2;
	add.s64 	%rd10, %rd4, %rd7;
	st.global.f32 	[%rd10], %f3;
	ret;
}
```

Above are the contents of vadd.ptx - unless you are an assembly code expert I suspect you will agree this is not very informative. PTX stands for Parallel Thread eXecution and we will briefly discuss two ways to make PTX code in this tutorial. The first way to make PTX code is by compiling CUDA C code. So what did the CUDA C code that made this PTX look like?

```
extern "C" {

__global__ void kernel_vadd(const float *a, const float *b, float *c)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

}
```

Even if you don't know C, hopefully you will be able to see that this function calculates a unique thread index (```i```) based on the block the thread is in and the thread's index within that block, then adds the values of ```a``` and ```b``` at that index and stores them in ```c```. This is very similar to many of the kernels we have written before. We can make PTX code from CUDA C by executing the following command in a terminal

```
nvcc --ptx /path/to/cuda_C_file.cu
```

It is also possible to make PTX code from Julia code using functions in CUDAnative such as ```code_ptx()``` - see http://juliagpu.github.io/CUDAnative.jl/latest/lib/reflection.html# CUDAnative.code_ptx for details.

Ok, so the outcome of all of the above is that we can now have a calleable GPU compatible function called ```vadd``` which performs vector addition. Next, we create our data:

```
# Make data
dims = 100
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)
```

This is done on a CPU and there is nothing really to note except that we are not using CuArrays. Next, we need to allocate memory on the GPU for ```a```, ```b``` and ```c```:

```
# Allocate memory for a and b on device stream 1
buf_a1 = Mem.alloc(sizeof(Float32))
buf_b1 = Mem.alloc(sizeof(Float32))
```

Again, this is the first time we have had to do this because previously it was taken care of for us by CUDAnative. In our streaming strategy, we are going to copy one element of ```a``` and ```b``` to each stream, calculate the value of the corresponding value of ```c```, store that result in ```c```, then copy the value of ```c``` back to the host. This means we need to allocate space for one element of ```a```, ```b``` and ```c``` for each stream. There are two streams, so in practice this means we need to allocate space twice. In the code above, we are allocating space for one element of ```a``` and one element of ```b``` for the first stream. Each element in ```a``` and ```b``` is a ```Float32```, so we use ```sizeof()``` to work out how many bytes we need to allocate. We also need to allocate space for ```a``` and ```b``` in stream 2, so we write:

```
# Allocate memory for a and b on device stream 2
buf_a2 = Mem.alloc(sizeof(Float32))
buf_b2 = Mem.alloc(sizeof(Float32))
```

Next, we allocate space for ```c``` on both streams and create the streams:

```
# Allocate memory for c on device
d_c1 = Mem.alloc(sizeof(Float32))
d_c2 = Mem.alloc(sizeof(Float32))

# Make streams
s1 = CuStream()
s2 = CuStream()
```

Now we are ready to actually start streaming data transfer and analysis. Like in the 'analysis only' streaming example, we do this here by iterating over our arrays in increments of size 2.

```
# Iterate over arrays in increments of 2
for i in 1:2:dims

    # Asynchronously copy a[i] and a[i+1] onto device
    Mem.upload!(buf_a1, Ref(a, i), sizeof(Float32), s1, async = true)
    Mem.upload!(buf_a2, Ref(a, i+1), sizeof(Float32), s2, async = true)

    # Asynchronously copy b[i] and b[i+1] onto device
    Mem.upload!(buf_b1, Ref(b, i), sizeof(Float32), s1, async = true)
    Mem.upload!(buf_b2, Ref(b, i+1), sizeof(Float32), s2, async = true)

    # Call vadd to run on gpu
    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a1, buf_b1, d_c1,
    threads = 1, stream = s1)
    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a2, buf_b2, d_c2,
    threads = 1, stream = s2)

    # Asynchronously copy c[i] and c[i+1] back to host
    Mem.download!(Ref(c, i), d_c1, sizeof(Float32), s1, async = true)
    Mem.download!(Ref(c, i+1), d_c2, sizeof(Float32), s2, async = true)


end
```

As before, we operate on the ```i```th element in one stream and the ```i + 1```th element in the other stream. In each stream, the order of operations is to copy (```Mem.upload!```) the value of the ```i```th or ```i + 1```th element of ```a``` and ```b``` from the host to the pre-allocated space on the device. Then we run the kernel (using ```cudacall```) and copy the result back to the host (```Mem.download!```).

There are several things worth noting here. First, we must explicitly specify ```async = true``` to ```Mem.upload!``` and ```Mem.download!``` if we want them to execute asynchronously - as a default these functions are synchronous. Second, note that we call the kernel using ```cudacall``` rather than ```@cuda``` - this is because ```@cuda``` is part of CUDAnative which does not support asynchronous copies. Thirdly, note that we must specify which stream we are using for each of these functions - otherwise all of these functions will execute on the same stream as a default.

Finally, we finish our script by checking the results of our calculation and destroying the device context:

```
# Check it worked
@test a+b ≈ c

# Destroy context
destroy!(ctx)
```

This example demonstrates that it is possible to stream data transfer and analysis in Julia. However, it should be noted that the code we wrote was so low level that almost every line we wrote was directly analogous to a CUDA C command. I would argue that the benefit of writing a program such as this in Julia versus C is debateable. There might be situations where it makes sense to write this type of program in Julia (for example, if you already had a huge Julia code base, a small part of which you wanted to port to GPU), but often it might actually be easier to use C.

In the next section, we will consider some Julia specific aspects of writing GPU compatible software.

# References

The data streaming and analysis example above was based on code taken from here: https://github.com/JuliaGPU/CUDAdrv.jl/tree/master/examples
