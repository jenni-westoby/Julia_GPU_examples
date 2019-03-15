#An Introduction to Parallelism

Congratulations! You (finally?) got your environment set up and are ready to start writing some GPU code. If this took you less than a week you should probably throw yourself a party.

##Party time

We're going to start our GPU adventure by considering a very simple program which adds two vectors together. If we wanted to do this on a CPU, we might write a function like this:

```
function add!(a,b,c)
    local tid = 1
    while (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
        tid += 1
    end
end
```

The function add! takes three vectors (a, b and c), adds each element of a and b together and stores the result in c. Note that we do not explicitly return c, because the exclamation mark at the end of add! indicates that add! is a function that modifies it's arguments.

We could call add! in a Julia script like this:

```
function main()

    #Make three vectors
    a = Vector{Any}(fill(undef, 10))
    b = Vector{Any}(fill(undef, 10))
    c = Vector{Any}(fill(undef, 10))

    #Fill a and b with values
    for i in 1:10
        a[i] = i
        b[i] = i * 2
    end

    #Fill c with values
    add!(a,b,c)

    #Do a sanity check
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end
end

main()
```

main() is a very simple function that makes three vectors, a, b and c. It populates a and b with values, calls add! to add each value in a and b together, then runs a for loop to check that the values stored in c make sense.

##Adding Vectors on a GPU

As exciting as the example above was, the eagle eyed amongst you may have noticed that it doesn't actually run on a GPU. Let's fix that.

The first thing we need to do is load packages that will enable us to run Julia code on GPUs.

```
using CuArrays, CUDAnative, CUDAdrv
```

CuArrays is a package that allows us to easily transfer arrays from CPU to GPU. CUDAnative allows us to write relatively high level code for executing functions on GPUs. We will not explicitly call CUDAdrv in our example, but much of CUDAnative depends on CUDAdrv to work.

Next, we need to identify what part of our example could benefit from being ported to GPU. Since most of the actual work is being done in add!, this is an obvious target. Let's modify add! so that it could be executed on a GPU.

```
function add!(a,b,c)
    local tid = 1
    while (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
        tid += 1
    end
    return nothing
end

```

Since add! is now ready to run on a GPU, we have thus transformed add! from an ordinary function to a kernel. Isn't unnecessary terminology wonderful?

Aside from now referring to add! as a kernel rather than a function, the only thing that has changed between the CPU and GPU version of add! is the addition of this line:

```
return nothing
```

CUDA requires that kernels must return nothing. Aside from meaning that we have to add this line to all of our kernels, this also means we potentially have to think a bit about how we will get the results of our GPU computations out of our kernels, since we can't directly ```return``` our results. As add! is a function which alters its arguments, this is not actually a problem which requires much thought in our example.

Let's see how main has changed in the GPU version of our example.

```
function main()

    #Make three CuArrays
    a = CuArrays.CuArray(fill(0, 10))
    b = CuArrays.CuArray(fill(0, 10))
    c = CuArrays.CuArray(fill(0, 10))

    #Fill a and b with values
    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    #Execute the kernel
    @cuda add!(a,b,c)

    #Copy a,b and c back from the device to the host
    a = Array(a)
    b = Array(b)
    c = Array(c)

    #Do a sanity check
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end
end

main()
```

main() is looking pretty different from the CPU version of our code. Let's work through it step by step.

```
function main()

    #Make three CuArrays
    a = CuArrays.CuArray(fill(0, 10))
    b = CuArrays.CuArray(fill(0, 10))
    c = CuArrays.CuArray(fill(0, 10))
```

Like in the CPU version of main, we start by making three arrays. However here, instead of making three standard Julia arrays, we make three CuArrays. CuArrays are GPU compatible arrays. For reasons we will gloss over here, ordinary Julia arrays would not work in our example. Fortunately, CuArrays are a subtype of AbstractArrays and can often be treated exactly the same way as a normal AbstractArray. Many standard array operations work out of the box on CuArrays, see https://github.com/JuliaGPU/CuArrays.jl for a list. If you are curious why we can't use a normal AbstractArray or Array here, see 'Further Considerations' for details.

The next step of main is virtually identical to the CPU version.

```
#Fill a and b with values
for i in 1:10
    a[i] = i
    b[i] = i * 2
end
```

The only thing to note here is that like I promised, we can treat a and b exactly like an ordinary AbstractArray here, no special syntax is required.

In the next step we actually execute the kernel:

```
#Execute the kernel
@cuda add!(a,b,c)
```

This looks remarkably similar to the CPU version of main at this step, especially when you consider that this line is responsible for executing add! on a different type of computing chip. The magic is contained in ```@cuda```. ```@cuda``` is part of the CUDAnative package, and behind the scenes is responsible for transforming the add! function into a form recognised and executed by the GPU.

If you are familiar with CUDA C or C++, you might be surprised that main does not include any step to copy a, b and c from the host (CPU) to the device (GPU). This is taken care of behind the scenes by CUDAnative and CuArrays. However, you do explicitly need to copy your CuArrays back from device (GPU) to host (CPU), which is what happens in the next step of main:

```
#Copy a,b and c back from the device to the host
a = Array(a)
b = Array(b)
c = Array(c)
```

The function responsible for copying your CuArrays from device to host is ```Array()```.

Finally, we do the same sanity check as in the CPU version of main and make a function call to main:

```
    #Do a sanity check
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end
end

main()
```

Note that in both versions of main, the sanity check is carried out on the host (CPU), not the device (GPU).

So we've written our first Julia script that will execute on a GPU! That's pretty cool. But again, the eagle eyed amongst you might be grumbling. Whilst our script does run on a GPU, there is absolutely no parallelism in it. In fact, it is likely that the GPU version of our script is actually slower than the CPU version, given that GPU processors are generally slower than CPU processors AND we had to copy a load of data from host to device and back again in the GPU version, which we didn't have to bother with in the CPU version. Time to introduce some parallelism to our script.

##Parallelising over threads

Let's see how main changes when we run the kernel over multiple threads:

```
function main()

    #Make three CuArrays
    a = CuArrays.CuArray(fill(0, 10))
    b = CuArrays.CuArray(fill(0, 10))
    c = CuArrays.CuArray(fill(0, 10))

    #Fill a and b with values
    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    #Execute the kernel
    @cuda threads=10 add!(a,b,c)

    #Copy a,b and c back from the device to the host
    a = Array(a)
    b = Array(b)
    c = Array(c)

    #Do a sanity check
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end
end

main()
```

The only line that has changed is this line:

```
@cuda threads=10 add!(a,b,c)
```

To make the kernel run on 10 threads, we have added the argument ```threads=10``` before our call to ```add!(a,b,c)```. That's it. However, let's think about what this will actually do. Running our current version of add! over 10 threads simply amounts to running add! 10 times simultaneously. Obviously, this will not be any faster than running add! once.

Let's modify add! so we can make a more productive use of the 10 threads.

```
function add!(a,b,c)
    tid = threadIdx().x
    if (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
    end
    return nothing
end
```

Again, we have only changed one line:

```
tid = threadIdx().x
```

```threadIdx()``` is a function from CUDAnative which returns the three dimensional index of the thread that this particular instance of the kernel is running on. Here, we use ```threadIdx().x``` to get the x coordinate for the thread the kernel is running on. As we specified in our call to ```@cuda``` that we would use 10 threads, the value of threadIdx().x will be between 1 and 10 for each instance of the kernel. Therefore, when we call

```
@cuda threads=10 add!(a,b,c)
```

We spawn 10 threads and on each thread one element of c is calculated. Clearly this will be a lot faster than calculating all 10 elements sequentially.


##Parallelising over blocks

You will recall from "Some Background on GPUs" that GPUs are composed of blocks of threads. In addition to parallelising our code over threads, we have the option of parallelising over blocks. To parallelise over blocks instead of threads, we change this line in main

```
@cuda threads=10 add!(a,b,c)
```

to

```
@cuda blocks=10 add!(a,b,c)
```

and change this line in add!

```
tid = threadIdx().x
```

to

```
tid = blockIdx().x
```

And now our code parallelises over blocks rather than threads! The complete script to parallelise over blocks is below:

```
function add!(a,b,c)
    tid = blockIdx().x
    if (tid <= min(length(a), length(b), length(c)))
        c[tid] = a[tid] + b[tid]
    end
    return nothing
end

function main()

    #Make three CuArrays
    a = CuArrays.CuArray(fill(0, 10))
    b = CuArrays.CuArray(fill(0, 10))
    c = CuArrays.CuArray(fill(0, 10))

    #Fill a and b with values
    for i in 1:10
        a[i] = -i
        b[i] = i * i
    end

    #Execute the kernel
    @cuda blocks=10 add!(a,b,c)

    #Copy a,b and c back from the device to the host
    a = Array(a)
    b = Array(b)
    c = Array(c)

    #Do a sanity check
    for i in 1:length(a)
        @test a[i] + b[i] ≈ c[i]
    end
end

main()
```

Congratulations, you have now written your first Julia scripts which parallelise over threads and blocks! We will move on to a more involved example in the next section.
