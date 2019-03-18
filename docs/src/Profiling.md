# Profiling

There are a number of reasons we might want to profile our GPU code, including:

- To work out which bits of our program take the most time to run.
- To check that our kernel is executing on as many threads and blocks as we expect.
- To check that our kernel is executing on as many streams as expected.

We can use a tool from the CUDA toolkit called ```nvprof``` for profiling. The following command can be used to call nvprof to profile a specific Julia GPU program from the command line:

```
$ nvprof --profile-from-start off /path/to/julia /path/to/julia/script
```

Note that we use ```--profile-from-start off``` to tell ```nvprof``` not to start profiling until we tell it to. There are several reasons we might not want to have ```nvprof``` start profiling straight away. One reason is that Julia is a Just-In-Time (JIT) compiled language. We often don't want to include the time taken to compile our scripts in our profiling estimates. Another reason is that GPU software usually executes some commands on CPU as well as GPU, and we may only want to profile commands executed on GPU. Finally, we might only be interested in a particular bit of our script, such as the kernel or the time taken to copy data from host to device. We can turn ```--profile-from-start``` off to enable us to profile just the bit we are interested.

# nvprof

Let's use our vector addition kernel streaming example to investigate how we can use ```nvprof``` for profiling. We have modified the example so it can be profiled with ```nvprof```:

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
CUDAdrv.@profile main()
```

The only change we have made to this example is to add this line at the end:

```
CUDAdrv.@profile main()
```

This line executes ```main()``` whilst activating the CUDA profiler. This signals to ```nvprof``` the point at which to start profiling. Notice that we have already called ```main()``` once in the line before we call ```CUDAdrv.@profile``` - this is to force ```main()``` to compile before we start profiling. If we don't call the function we want to profile once before we start profiling then our profiling estimates will include time spent compiling.

Having modified our example script to invoke the ```nvprof``` profiler, we can now get profiling information by executing the following on the command line:

```
$ nvprof --profile-from-start off julia gpu_vector_sums_streams.jl
```

Where ```gpu_vector_sums_streams.jl``` is the name of our example script. The output of this command should look something like this:

```
Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   50.18%  21.632us        23     940ns     928ns  1.2160us  [CUDA memcpy HtoD]
       39.64%  17.088us        10  1.7080us  1.5680us  2.1440us  ptxcall_add__1
       10.17%  4.3850us         3  1.4610us  1.3440us  1.5680us  [CUDA memcpy DtoH]
API calls:   82.95%  1.5925ms        10  159.25us  5.0680us  1.5383ms  cuLaunchKernel
       10.24%  196.67us        23  8.5500us  6.6030us  26.618us  cuMemcpyHtoD
        2.34%  44.868us         3  14.956us  8.3740us  26.289us  cuMemAlloc
        2.25%  43.195us         3  14.398us  12.429us  17.862us  cuMemcpyDtoH
        1.70%  32.641us         2  16.320us  8.3680us  24.273us  cuStreamCreate
        0.39%  7.4110us        15     494ns     426ns     672ns  cuCtxGetCurrent
        0.13%  2.4880us         1  2.4880us  2.4880us  2.4880us  cuDeviceGetCount
```

There is some pretty interesting information in this output - for example, the first line of the output tells us that 50.18% of GPU time is spent copying data from host to device. This is a good illustration of how slow copying data between CPU and GPU often is. We can get some more interesting information by adding an extra flag to our ```nvprof``` command:

```
$ nvprof --profile-from-start off --print-gpu-trace julia gpu_vector_sums_streams.jl
```

The output of this command should look something like this:

```
Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
15.1992s  1.2480us                    -               -         -         -         -       80B  61.133MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1992s  1.1200us                    -               -         -         -         -       80B  68.120MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1992s     928ns                    -               -         -         -         -       80B  82.213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1992s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1992s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1992s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1992s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1992s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     960ns                    -               -         -         -         -        8B  7.9473MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1993s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1994s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1994s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.1994s     928ns                    -               -         -         -         -        8B  8.2213MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
15.2010s  2.2720us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        16  ptxcall_add__1 [108]
15.2010s  1.9520us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        17  ptxcall_add__1 [110]
15.2010s  1.5680us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        16  ptxcall_add__1 [112]
15.2010s  1.5680us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        17  ptxcall_add__1 [114]
15.2010s  1.5680us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        16  ptxcall_add__1 [116]
15.2010s  1.5680us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        17  ptxcall_add__1 [118]
15.2010s  1.5680us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        16  ptxcall_add__1 [120]
15.2010s  1.6000us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        17  ptxcall_add__1 [122]
15.2010s  1.8880us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        16  ptxcall_add__1 [124]
15.2010s  1.5680us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        17  ptxcall_add__1 [126]
15.2010s  1.1840us                    -               -         -         -         -       80B  64.437MB/s      Device    Pageable  Quadro M1000M (         1         7  [CUDA memcpy DtoH]
15.2011s  1.1840us                    -               -         -         -         -       80B  64.437MB/s      Device    Pageable  Quadro M1000M (         1         7  [CUDA memcpy DtoH]
15.2011s  1.3120us                    -               -         -         -         -       80B  58.151MB/s      Device    Pageable  Quadro M1000M (         1         7  [CUDA memcpy DtoH]
```

Let's talk through what this output means. First there are a series of lines that look something like this:

```
Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
15.1992s  1.2480us                    -               -         -         -         -       80B  61.133MB/s    Pageable      Device  Quadro M1000M (         1         7  [CUDA memcpy HtoD]
```

These lines are describing the process of copying data from host to device (```[CUDA memcpy HtoD]```). We can see this is happening in stream 7. Next we see lines that look like these:

```
Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
15.2010s  2.2720us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        16  ptxcall_add__1 [108]
15.2010s  1.9520us              (1 1 1)         (1 1 1)        32        0B        0B         -           -           -           -  Quadro M1000M (         1        17  ptxcall_add__1 [110]
```

These lines are executing the kernel. There are two things to notice here:

1. The Grid Size and Block Size are both ```(1 1 1)```, indicating that the kernel is executing sequentially. If you look back at the example script you will see that this is as expected - we always call the kernel on 1 thread and 1 block (the default number of threads and blocks). If you specified more threads or blocks to ```@cuda``` these fields would change.

2. The first line executes in stream 16 and the second executes in stream 17. Again, if you look back at the example script this is as expected - we create two streams then execute our kernel alternately in each stream.

Finally, there are some lines copying data back from device to host:

```
Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
15.2010s  1.1840us                    -               -         -         -         -       80B  64.437MB/s      Device    Pageable  Quadro M1000M (         1         7  [CUDA memcpy DtoH]
```

This occurs in stream 7 again.

# Zoom in on the kernel

One of the main take homes of the profiling above should be that most of the execution time is taken up copying data. Let's imagine that we decided we weren't interested in profiling data transfer and just wanted to profile the kernel. We could modify our example script so it looked like below:

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
        CUDAdrv.@profile @cuda threads = 1 stream = s1 add!(a,b,c, i)
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

Notice that we no longer use ```CUDAdrv.@profile``` to profile ```main()```, but we now use it to profile our call to ```@cuda threads = 1 stream = s1 add!(a,b,c, i)```. Now we call

```
$ nvprof --profile-from-start off julia gpu_vector_sums_streams.jl
```

from the command line, and get the following output:

```
Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:  100.00%  10.304us         5  2.0600us  1.8560us  2.3040us  ptxcall_add__1
API calls:   75.82%  11.778ms         5  2.3556ms  1.4952ms  2.6888ms  cuLaunchKernel
       24.17%  3.7540ms         5  750.79us     864ns  961.91us  cuCtxGetCurrent
        0.01%  2.1640us         1  2.1640us  2.1640us  2.1640us  cuDeviceGetCount
```

Since we are just profiling the kernel now, unsurprisingly we find that 100% of the profiled GPU time was spent executing the kernel. The output to

```
$ nvprof --profile-from-start off --print-gpu-trace julia gpu_vector_sums_streams.jl
```

is shown below:

```
Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream  Name
14.9673s  2.0800us              (1 1 1)         (1 1 1)        32        0B        0B  Quadro M1000M (         1        14  ptxcall_add__1 [54]
16.9861s  1.7920us              (1 1 1)         (1 1 1)        32        0B        0B  Quadro M1000M (         1        14  ptxcall_add__1 [62]
19.0047s  2.0480us              (1 1 1)         (1 1 1)        32        0B        0B  Quadro M1000M (         1        14  ptxcall_add__1 [70]
21.0233s  1.8560us              (1 1 1)         (1 1 1)        32        0B        0B  Quadro M1000M (         1        14  ptxcall_add__1 [78]
23.0424s  1.8880us              (1 1 1)         (1 1 1)        32        0B        0B  Quadro M1000M (         1        14  ptxcall_add__1 [86]
```

Again, all of the data transfer steps have vanished from the profiling output, since we have only profiled kernel execution. Notice that we see 5 lines of output here, corresponding to 5 calls of the kernel, and that the kernel is always called in the same stream, because we only profiled calls to the kernel in stream ```s1```.

Hopefully this section has shown some of the methods you can use to profile GPU software that you write in Julia. In the next section, we will discuss things to be aware of when thinking about performance.
