var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Learn-to-Develop-GPU-Software-Using-Julia-1",
    "page": "Home",
    "title": "Learn to Develop GPU Software Using Julia",
    "category": "section",
    "text": "A number of packages have been developed for Julia that enable developers to write Julia code that will run on GPUs. But until now, open source documentation which would enable a developer to go from zero CUDA knowledge to making effective use of these packages has not existed. This tutorial aims to change that. In this tutorial, I will guide you through the full workflow of writing a basic program in Julia that will run on GPU. Things you will learn from this tutorial include:What a GPU is and why writing code to run on GPUs might be useful.\nHow to set up your computer and install the graphics drivers necessary to make use of your GPU.\nHow to write Julia code that will run on a GPU in a parallel manner using threads, blocks, shared memory and streaming.\nHow to benchmark Julia code that runs on GPUs.This tutorial assumes you already know how to write code in Julia. Julia is a high level, syntactically intuitive language, so if you don\'t know Julia but do know other programming languages you may find you are able to follow along anyway. If you would like to get started coding in Julia, I recommend reading some of the Julia Documentation: https://docs.julialang.org/en/v1.0/.This tutorial is intended to provide a foundation for developers to start writing Julia code that runs on GPUs. It is not intended to be a comprehensive guide to everything you could conceivably do on a GPU using Julia. Hopefully, you will find it instead gives a gentle introduction to some core concepts in GPU programming and how you can use them in Julia. I focus on the CUDAdrv, CUDAnative and CuArrays packages in this tutorial, although other Julia packages for using GPUs exist - see https://github.com/JuliaGPU for details."
},

{
    "location": "GPU_background/#",
    "page": "Some Background on GPUs",
    "title": "Some Background on GPUs",
    "category": "page",
    "text": ""
},

{
    "location": "GPU_background/#Some-Background-on-GPUs-1",
    "page": "Some Background on GPUs",
    "title": "Some Background on GPUs",
    "category": "section",
    "text": "This section gives a very light overview of what GPUs are and some relevant information on how they are organised for developers."
},

{
    "location": "GPU_background/#What-are-GPUs?-1",
    "page": "Some Background on GPUs",
    "title": "What are GPUs?",
    "category": "section",
    "text": "You have probably heard of a type of computer chip called a CPU. GPUs are a different type of computer chip. The major difference between CPUs and GPUs is how each chip is organised. Whereas CPUs tend to have a small number of threads, GPUs can have orders of magnitudes more. On GPUs, you can write code that parallelises across thousands of threads, something that simply would not be possible on CPUs without great effort and expense."
},

{
    "location": "GPU_background/#How-are-GPUs-organised?-1",
    "page": "Some Background on GPUs",
    "title": "How are GPUs organised?",
    "category": "section",
    "text": "(Image: )Each GPU contains a \'grid\' of \'blocks\'. Each block in that grid is itself composed of a matrix of \'threads\'. Programs written to run on GPUs can be parallelised over blocks or threads or both. Each block has an index representing its position in the grid, and each thread has an index representing its position within its block."
},

{
    "location": "GPU_background/#Host-and-Device-1",
    "page": "Some Background on GPUs",
    "title": "Host and Device",
    "category": "section",
    "text": "Software written for GPUs rarely runs entirely on GPUs. Usually, at least a small part of GPU compatible software will run on CPUs. For example, copying data from CPUs to GPUs is a required step for most GPU software, and the copying instruction will need to execute at least partly on a CPU. Because GPU software typically executes instructions on both CPUs and GPUs, it is useful to have some terminology to describe where a particular line of code is being executed. Code that executes on CPUs is commonly described as executing \'on the host\', whilst code executing on GPUs is described as executing \'on the device\'. We can also use the terms \'host\' and \'device\' to describe where data lives in memory and operations copying data between CPUs and GPUs."
},

{
    "location": "GPU_background/#Kernel-1",
    "page": "Some Background on GPUs",
    "title": "Kernel",
    "category": "section",
    "text": "A common way to organise code that will run on a GPU is to write a function that will execute on the GPU. This function is typically called after all the data required by the function has been copied from host (CPU) to device (GPU). Because the function is typically executed thousands of times in parallel, the function that will execute on the GPU is given the special name of \'kernel\'. "
},

{
    "location": "Accessing_GPUs/#",
    "page": "Accessing GPUs",
    "title": "Accessing GPUs",
    "category": "page",
    "text": ""
},

{
    "location": "Accessing_GPUs/#Accessing-GPUs-1",
    "page": "Accessing GPUs",
    "title": "Accessing GPUs",
    "category": "section",
    "text": "You basically have three options for accessing GPUs:Access GPUs on a commercial cloud based platform eg. AWS\nAccess GPUs using a HPC you have access to through your company/academic institution.\nUse the GPU in your own computer"
},

{
    "location": "Accessing_GPUs/#Option-1-1",
    "page": "Accessing GPUs",
    "title": "Option 1",
    "category": "section",
    "text": "If you are going for option 1, the cloud based platform you are using should have instructions for how to gain access to the GPUs you are paying for. It is quite likely that the cloud based platform will already have the CUDA toolkit installed, so once you have managed to log in you are probably good to go!"
},

{
    "location": "Accessing_GPUs/#Option-2-1",
    "page": "Accessing GPUs",
    "title": "Option 2",
    "category": "section",
    "text": "If you are going for option 2, talk to the HPC administrator to find out how to get access to the GPUs and check that the CUDA toolkit are already installed."
},

{
    "location": "Accessing_GPUs/#Option-3-1",
    "page": "Accessing GPUs",
    "title": "Option 3",
    "category": "section",
    "text": "If you are going for option 3, things are about to get interesting.These are some instructions you can try following to get the GPU in your computer working in a way that Julia will be able to interact with it. I make no promises that they will work and will not offer support if they fail for you. These instructions assume you are working in a Linux environment. I don\'t know how to make your GPU work in a Windows or Mac environment.Step 1: Check that your computer has a GPU.Step 2: Check that your computer has a GPU that is actually supported by NVIDIA (CUDA is not supported on old GPUs).Step 3: Replace your operating system with a fresh install of Ubuntu 18.04. No, I am not joking. Remember to back up your files before you install a new operating system just in case something goes wrong.Step 4: Open a terminal and type in the following commands:sudo add-apt-repository ppa:graphics-drivers/ppa\n\nsudo apt update\n\nsudo ubuntu-drivers autoinstall\nStep 5: Reboot your computerStep 6: Open a terminal and type in the following:sudo apt install nvidia-cuda-toolkit gcc-6\n\nnvcc --version\nIf your install has worked, the final command should print out some version information.Step 7: Assuming all of the above worked, install the latest version of Julia and continue working through this tutorial.Disclaimer The above are NOT authoritative instructions on how to install the CUDA toolkit, they are instructions that worked for me. You should visit the NVIDIA website for authoritative instructions.Disclaimer I am writing these instructions in spring 2019. If you are reading this disclaimer and the year is 2021 or greater, these instructions may be woefully out of date and better ignored."
},

{
    "location": "Accessing_GPUs/#Surely-I-don\'t-really-need-to-install-a-new-operating-system-1",
    "page": "Accessing GPUs",
    "title": "Surely I don\'t really need to install a new operating system",
    "category": "section",
    "text": "The CUDA toolkit and drivers are notoriously difficult to install. It might sound a bit mad, but installing a fresh version of Ubuntu 18.04 is the most reliable method for installing CUDA that I have found. Note, steps 4-6 are taken from the second answer here: https://askubuntu.com/questions/1028830/how-do-i-install-cuda-on-ubuntu-18-04, so a big thank you to eromod and N0rbert."
},

{
    "location": "Setup_Julia/#",
    "page": "Set-up Julia",
    "title": "Set-up Julia",
    "category": "page",
    "text": ""
},

{
    "location": "Setup_Julia/#Set-up-Julia-1",
    "page": "Set-up Julia",
    "title": "Set-up Julia",
    "category": "section",
    "text": "This section assumes that you have already carried out the instructions in \'Accessing GPUs\' and that you now have access to a machine whichContains a CUDA-compatible GPU, and\nHas a working CUDA toolkit installation.If this is not the case, many of the below steps will not work because they depend on the two conditions above being true.Assuming you did follow the instructions in \"Accessing GPUs\" and you have access to a computer with a GPU and a working CUDA toolkit installation, carry out the steps below to get a Julia environment that you can run this tutorial in.Step 1: Follow the link instructions to install Julia 1.0.3 if you haven\'t already done so: https://julialang.org/downloads/. I recommend against using your OS\'s package manager to install Julia, you are unlikely to get the correct version.Step 2 Within Julia, install CUDAdrv, CUDAnative and CuArrays using the commands below:]\nadd CUDAdrv\nadd CUDAnative\nadd CuArraysNote - this step is likely to fail if you do not have a working CUDA installation."
},

{
    "location": "Setup_Julia/#Common-Problems-1",
    "page": "Set-up Julia",
    "title": "Common Problems",
    "category": "section",
    "text": "I\'m sure I installed the CUDA toolkit correctly yesterday but now nothing is working!Try rebooting your machine. Sometimes when your machine goes to sleep/ you close your laptop lid/ whatever, the graphics driver configuration is slightly different when it wakes up again. Rebooting your machine fixes this problem.I installed the packages in a slightly different order to you and they didn\'t install properly.Try uninstalling the packages and re-installing them in the same order as me.It turns out there is something wrong with my CUDA installation.This is a very common problem. You could try following the instructions in \"Set up your GPU\" to install the CUDA toolkit. If none of the options there work for you then I\'m afraid I am out of suggestions :("
},

{
    "location": "Vector_addition/#",
    "page": "An Introduction to Parallelism",
    "title": "An Introduction to Parallelism",
    "category": "page",
    "text": ""
},

{
    "location": "Vector_addition/#An-Introduction-to-Parallelism-1",
    "page": "An Introduction to Parallelism",
    "title": "An Introduction to Parallelism",
    "category": "section",
    "text": "Congratulations! You (finally?) got your environment set up and are ready to start writing some GPU code. If this took you less than a week you should probably throw yourself a party."
},

{
    "location": "Vector_addition/#Party-time-1",
    "page": "An Introduction to Parallelism",
    "title": "Party time",
    "category": "section",
    "text": "We\'re going to start our GPU adventure by considering a very simple program which adds two vectors together. If we wanted to do this on a CPU, we might write a function like this:function add!(a,b,c)\n    local tid = 1\n    while (tid <= min(length(a), length(b), length(c)))\n        c[tid] = a[tid] + b[tid]\n        tid += 1\n    end\nendThe function add!() takes three vectors (a, b  and c), adds each element of a and b together and stores the result in c. Note that we do not explicitly return c, because the exclamation mark at the end of add!() indicates that add!() is a function that modifies it\'s arguments.We could call add! in a Julia script like this:function main()\n\n    # Make three vectors\n    a = Vector{Any}(fill(undef, 10))\n    b = Vector{Any}(fill(undef, 10))\n    c = Vector{Any}(fill(undef, 10))\n\n    # Fill a and b with values\n    for i in 1:10\n        a[i] = i\n        b[i] = i * 2\n    end\n\n    # Fill c with values\n    add!(a,b,c)\n\n    # Do a sanity check\n    for i in 1:length(a)\n        @test a[i] + b[i] ≈ c[i]\n    end\nend\n\nmain()main() is a very simple function that makes three vectors, a, b and c. It populates a and b with values, calls add!() to add each value in a and b together, then runs a for loop to check that the values stored in c make sense."
},

{
    "location": "Vector_addition/#Adding-Vectors-on-a-GPU-1",
    "page": "An Introduction to Parallelism",
    "title": "Adding Vectors on a GPU",
    "category": "section",
    "text": "As exciting as the example above was, the eagle eyed amongst you may have noticed that it doesn\'t actually run on a GPU. Let\'s fix that.The first thing we need to do is load packages that will enable us to run Julia code on GPUs.using CuArrays, CUDAnative, CUDAdrvCuArrays is a package that allows us to easily transfer arrays from CPU to GPU. CUDAnative allows us to write relatively high level code for executing functions on GPUs. We will not explicitly call CUDAdrv in our example, but much of CUDAnative depends on CUDAdrv to work.Next, we need to identify what part of our example could benefit from being ported to GPU. Since most of the actual work is being done in add!, this is an obvious target. Let\'s modify add! so that it could be executed on a GPU.function add!(a,b,c)\n    local tid = 1\n    while (tid <= min(length(a), length(b), length(c)))\n        c[tid] = a[tid] + b[tid]\n        tid += 1\n    end\n    return nothing\nend\nSince add! is now ready to run on a GPU, we have thus transformed add! from an ordinary function to a kernel. Isn\'t unnecessary terminology wonderful?Aside from now referring to add! as a kernel rather than a function, the only thing that has changed between the CPU and GPU version of add! is the addition of this line:return nothingCUDA requires that kernels must return nothing. Aside from meaning that we have to add this line to all of our kernels, this also means we potentially have to think a bit about how we will get the results of our GPU computations out of our kernels, since we can\'t directly return our results. As add! is a function which alters its arguments, this is not actually a problem which requires much thought in our example.Let\'s see how main has changed in the GPU version of our example.function main()\n\n    # Make three CuArrays\n    a = CuArrays.CuArray(fill(0, 10))\n    b = CuArrays.CuArray(fill(0, 10))\n    c = CuArrays.CuArray(fill(0, 10))\n\n    # Fill a and b with values\n    for i in 1:10\n        a[i] = -i\n        b[i] = i * i\n    end\n\n    # Execute the kernel\n    @cuda add!(a,b,c)\n\n    # Copy a,b and c back from the device to the host\n    a = Array(a)\n    b = Array(b)\n    c = Array(c)\n\n    # Do a sanity check\n    for i in 1:length(a)\n        @test a[i] + b[i] ≈ c[i]\n    end\nend\n\nmain()main() is looking pretty different from the CPU version of our code. Let\'s work through it step by step.function main()\n\n    # Make three CuArrays\n    a = CuArrays.CuArray(fill(0, 10))\n    b = CuArrays.CuArray(fill(0, 10))\n    c = CuArrays.CuArray(fill(0, 10))Like in the CPU version of main, we start by making three arrays. However here, instead of making three standard Julia arrays, we make three CuArrays. CuArrays are GPU compatible arrays. For reasons we will gloss over here, ordinary Julia arrays would not work in our example. Fortunately, CuArrays are a subtype of AbstractArrays and can often be treated exactly the same way as a normal AbstractArray. Many standard array operations work out of the box on CuArrays, see https://github.com/JuliaGPU/CuArrays.jl for a list. If you are curious why we can\'t use a normal AbstractArray or Array here, see \'Further Considerations\' for details.The next step of main is virtually identical to the CPU version.# Fill a and b with values\nfor i in 1:10\n    a[i] = i\n    b[i] = i * 2\nendThe only thing to note here is that like I promised, we can treat a and b exactly like an ordinary AbstractArray here, no special syntax is required.In the next step we actually execute the kernel:# Execute the kernel\n@cuda add!(a,b,c)This looks remarkably similar to the CPU version of main at this step, especially when you consider that this line is responsible for executing add! on a different type of computing chip. The magic is contained in @cuda. @cuda is part of the CUDAnative package, and behind the scenes is responsible for transforming the add! function into a form recognised and executed by the GPU.If you are familiar with CUDA C or C++, you might be surprised that main does not include any step to copy a, b and c from the host (CPU) to the device (GPU). This is taken care of behind the scenes by CUDAnative and CuArrays. However, you do explicitly need to copy your CuArrays back from device (GPU) to host (CPU), which is what happens in the next step of main:# Copy a,b and c back from the device to the host\na = Array(a)\nb = Array(b)\nc = Array(c)The function responsible for copying your CuArrays from device to host is Array().Finally, we do the same sanity check as in the CPU version of main and make a function call to main:    # Do a sanity check\n    for i in 1:length(a)\n        @test a[i] + b[i] ≈ c[i]\n    end\nend\n\nmain()Note that in both versions of main, the sanity check is carried out on the host (CPU), not the device (GPU).So we\'ve written our first Julia script that will execute on a GPU! That\'s pretty cool. But again, the eagle eyed amongst you might be grumbling. Whilst our script does run on a GPU, there is absolutely no parallelism in it. In fact, it is likely that the GPU version of our script is actually slower than the CPU version, given that GPU processors are generally slower than CPU processors AND we had to copy a load of data from host to device and back again in the GPU version, which we didn\'t have to bother with in the CPU version. Time to introduce some parallelism to our script."
},

{
    "location": "Vector_addition/#Parallelising-over-threads-1",
    "page": "An Introduction to Parallelism",
    "title": "Parallelising over threads",
    "category": "section",
    "text": "Let\'s see how main changes when we run the kernel over multiple threads:function main()\n\n    # Make three CuArrays\n    a = CuArrays.CuArray(fill(0, 10))\n    b = CuArrays.CuArray(fill(0, 10))\n    c = CuArrays.CuArray(fill(0, 10))\n\n    # Fill a and b with values\n    for i in 1:10\n        a[i] = -i\n        b[i] = i * i\n    end\n\n    # Execute the kernel\n    @cuda threads=10 add!(a,b,c)\n\n    # Copy a,b and c back from the device to the host\n    a = Array(a)\n    b = Array(b)\n    c = Array(c)\n\n    # Do a sanity check\n    for i in 1:length(a)\n        @test a[i] + b[i] ≈ c[i]\n    end\nend\n\nmain()The only line that has changed is this line:@cuda threads=10 add!(a,b,c)To make the kernel run on 10 threads, we have added the argument threads=10 before our call to add!(a,b,c). That\'s it. However, let\'s think about what this will actually do. Running our current version of add! over 10 threads simply amounts to running add! 10 times simultaneously. Obviously, this will not be any faster than running add! once.Let\'s modify add! so we can make a more productive use of the 10 threads.function add!(a,b,c)\n    tid = threadIdx().x\n    if (tid <= min(length(a), length(b), length(c)))\n        c[tid] = a[tid] + b[tid]\n    end\n    return nothing\nendAgain, we have only changed one line:tid = threadIdx().xthreadIdx() is a function from CUDAnative which returns the three dimensional index of the thread that this particular instance of the kernel is running on. Here, we use threadIdx().x to get the x coordinate for the thread the kernel is running on. As we specified in our call to @cuda that we would use 10 threads, the value of threadIdx().x will be between 1 and 10 for each instance of the kernel. Therefore, when we call@cuda threads=10 add!(a,b,c)We spawn 10 threads and on each thread one element of c is calculated. Clearly this will be a lot faster than calculating all 10 elements sequentially."
},

{
    "location": "Vector_addition/#Parallelising-over-blocks-1",
    "page": "An Introduction to Parallelism",
    "title": "Parallelising over blocks",
    "category": "section",
    "text": "You will recall from \"Some Background on GPUs\" that GPUs are composed of blocks of threads. In addition to parallelising our code over threads, we have the option of parallelising over blocks. To parallelise over blocks instead of threads, we change this line in main@cuda threads=10 add!(a,b,c)to@cuda blocks=10 add!(a,b,c)and change this line in add!tid = threadIdx().xtotid = blockIdx().xAnd now our code parallelises over blocks rather than threads! The complete script to parallelise over blocks is below:function add!(a,b,c)\n    tid = blockIdx().x\n    if (tid <= min(length(a), length(b), length(c)))\n        c[tid] = a[tid] + b[tid]\n    end\n    return nothing\nend\n\nfunction main()\n\n    # Make three CuArrays\n    a = CuArrays.CuArray(fill(0, 10))\n    b = CuArrays.CuArray(fill(0, 10))\n    c = CuArrays.CuArray(fill(0, 10))\n\n    # Fill a and b with values\n    for i in 1:10\n        a[i] = -i\n        b[i] = i * i\n    end\n\n    # Execute the kernel\n    @cuda blocks=10 add!(a,b,c)\n\n    # Copy a,b and c back from the device to the host\n    a = Array(a)\n    b = Array(b)\n    c = Array(c)\n\n    # Do a sanity check\n    for i in 1:length(a)\n        @test a[i] + b[i] ≈ c[i]\n    end\nend\n\nmain()Congratulations, you have now written your first Julia scripts which parallelise over threads and blocks! We will move on to a more involved example in the next section."
},

{
    "location": "Vector_dot_product/#",
    "page": "Shared Memory and Synchronisation",
    "title": "Shared Memory and Synchronisation",
    "category": "page",
    "text": ""
},

{
    "location": "Vector_dot_product/#Shared-Memory-and-Synchronisation-1",
    "page": "Shared Memory and Synchronisation",
    "title": "Shared Memory and Synchronisation",
    "category": "section",
    "text": "Following our example of vector addition in the previous section, you may be left wondering what the point of making a distinction between blocks and threads is. This section should make this clear.You may recall from \"Some Background on GPUs\" that GPUs are composed of grids of blocks, where each block contains threads.(Image: )In addition to threads, each block contains \'shared memory\'. Shared memory is memory which can be read and written to by all the threads in a given block. Shared memory can\'t be accessed by threads not in the specified block. This is illustrated in the diagram below.(Image: )In the code we wrote for vector addition, we did not use shared memory. Instead we used global memory. Global memory can be accessed from all threads, regardless of what block they live in, but has the disadvantage of taking a lot longer to read from compared with shared memory. There are two main reasons we might use shared memory in a program:It can be useful to have threads which can \'communicate\' with each other via shared memory.\nIf we have a kernel that frequently has to read from memory, it might be quicker to have it read from shared rather than global memory (but this very much depends on your particular algorithm).Of course, there is an obvious potential disadvantage to using shared memory. Giving multiple threads the capability to read and write from the same memory is potentially powerful. However it is also potentially dangerous. Now it is possible for threads to try to write to the same location in memory simultaneously. If we want there to be a dependency between threads, where thread A reads the results written by thread B, there is no automatic guarantee that thread A will not try to read the results before thread B has written them. We need a method to synchronise threads so this type of situation can be avoided. Fortunately, such a method exists as part of CUDAnative."
},

{
    "location": "Vector_dot_product/#Vector-Dot-Product-1",
    "page": "Shared Memory and Synchronisation",
    "title": "Vector Dot Product",
    "category": "section",
    "text": "We will use a vector dot product to explore some of the ideas introduced above. A vector dot product is when each of the elements of a vector is multiplied by the corresponding element in a second vector. Then, all of the multiplied elements are added together to give a single number as a result.As before, we begin our script by loading the Julia packages we need to write GPU compatible code.using CuArrays, CUDAnative, CUDAdrvNext, we need to write the kernel. It is a lot to take in, but don\'t worry, we will go through it step by step.function dot(a,b,c, N, threadsPerBlock, blocksPerGrid)\n\n    # Set up shared memory cache for this current block.\n    cache = @cuDynamicSharedMem(Int64, threadsPerBlock)\n\n    # Initialise some variables.\n    tid = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x\n    cacheIndex = threadIdx().x - 1\n    temp::Int64 = 0\n\n    # Iterate over vector to do dot product in parallel way\n    while tid < N\n        temp += a[tid + 1] * b[tid + 1]\n        tid += blockDim().x * gridDim().x\n    end\n\n    # set cache values\n    cache[cacheIndex + 1] = temp\n\n    # synchronise threads\n    sync_threads()\n\n    # In the step below, we add up all of the values stored in the cache\n    i::Int = blockDim().x/2\n    while i!=0\n        if cacheIndex < i\n            cache[cacheIndex + 1] += cache[cacheIndex + i + 1]\n        end\n        sync_threads()\n        i/=2\n    end\n\n    # cache[1] now contains the sum of vector dot product calculations done in\n    # this block, so we write it to c\n    if cacheIndex == 0\n        c[blockIdx().x] = cache[1]\n    end\n\n    return nothing\nendThis is more complicated than the vector addition kernel, so let\'s work through it bit by bit. Let\'s start by focusing on the lines below:function dot(a,b,c, N, threadsPerBlock, blocksPerGrid)\n\n    # Set up shared memory cache for this current block.\n    cache = @cuDynamicSharedMem(Int64, threadsPerBlock)Here, we are setting a variable called cache to the output of a function call to @cuDynamicSharedMem. As the comment suggests, this is required to create a cache of shared memory that can be accessed by all the threads in the current block. @cuDynamicSharedMem is a function from CUDAnative which allocates an array in dynamic shared memory on the GPU. The first argument specifies the type of elements in the array and the second argument specifies the dimensions of the array. Socache = @cuDynamicSharedMem(Int64, threadsPerBlock)allocates an array in shared memory with the dimensions threadsPerBlock, where each element in the array is of type Int64.So now we have an array of size threadsPerBlock in shared memory which we can fill with Int64s. Next we set the value of the thread index (tid):# Initialise some variables.\ntid = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().xThis is the first time we\'ve mixed up thread and block indexes in the same kernel! So what\'s going on?The aim of this line of code is to generate a unique thread index for each thread. threadIdx().x gives the index for the current thread inside the current block. So threadIdx().x is not sufficient by itself because we are launching the kernel over multiple blocks. Each block has a thread with the index 1 (so threadIdx().x = 1), a second thread with the index 2 (threadIdx().x = 2) and so on, so we need a different approach to generate a unique thread index. blockDim().x gives number of threads in a block, which is the same for each block in a GPU. By multiplying the block index (blockIdx().x) and the number of threads in a block (blockDim().x), we count the threads in all the blocks before the one we are currently in. Then we add the thread index (threadIdx().x) in the current block to this total, thus generating a unique thread index for each thread across all blocks. This approach is illustrated below.(Image: )A final thing to note is that we subtract one from threadIdx().x and blockIdx().x. This is because Julia is tragically a one indexed programming language. You will notice a lot of plus and minus ones in this example, they are all there for this reason and whilst you are getting your head around the core concepts you should do you best to ignore them.Fortunately the next two lines are conceptually a lot simpler:cacheIndex = threadIdx().x - 1\ntemp::Int64 = 0cacheIndex is the index we will use to write an element to the array of shared memory we created. Remember shared memory is only accessible within the current block, so we do not need to worry about making a unique index across blocks like we did for tid. We set it to threadIdx().x - 1 so that each thread is writing to a separate location in shared memory - otherwise threads could overwrite the results calculated by other threads.Now we are ready to start calculating the dot product:# Iterate over vector to do dot product in parallel way\nwhile tid < N\n    temp += a[tid + 1] * b[tid + 1]\n    tid += blockDim().x * gridDim().x\nendFor context, N is the number of elements in a (which is the same as the number of elements in b). So while tid less than the number of elements in a, we increment the value of temp by the product of a[tid + 1] and b[tid + 1] - this is the core operation in a vector dot product. Then, we increment tid by the number of threads in a block (blockDim().x) times the number of blocks in a grid (gridDim().x), which is the total number of threads on the GPU. This line enables us to carry out dot products for vectors which have more elements than the total number of threads on our GPU.After exiting the while loop, we write the value calculated in temp to shared memory:# set cache values\ncache[cacheIndex + 1] = tempIn the next step of the kernel, we want to sum up all the values stored in shared memory. We do this by finding the sum of all the elements in cache. But remember that each thread is running asynchronously - just because one thread has finished executing the line:cache[cacheIndex + 1] = tempDoesn\'t mean that all threads have executed that line. To avoid trying to sum the elements of cache before they have all been written, we need to make the threads all pause and wait until every thread has reached the same line in the kernel. Fortunately, such a function exists as part of CUDAnative:# synchronise threads\nsync_threads()When each thread reaches this line, it pauses in its execution of the kernel until all of the threads in that block have reached the same place. Then, the threads restart again.Now all the threads have written to shared memory, we are ready to sum the elements of cache:# In the step below, we add up all of the values stored in the cache\ni::Int = blockDim().x/2\nwhile i!=0\n    if cacheIndex < i\n        cache[cacheIndex + 1] += cache[cacheIndex + i + 1]\n    end\n    sync_threads()\n    i/=2\nendHere, we initialise i as half of the total number of threads in a block. In the first iteration of the while loop, if cacheIndex is less than this number, we add the value stored at cache[cacheIndex + i + 1] to the value of cache[cacheIndex + 1]. Then we synchronise the threads again, divide i by two and enter the second while loop iteration. If you work through this conceptually, you should see that provided the number of threads in a block is an even number, eventually the value at cache[1] will be equal to the sum of all the elements in cache.Now we need to write the value of cache[1] to c (remember that we can not directly return the value of cache[1] due to the requirement that the kernel must always return nothing).# cache[1] now contains the sum of vector dot product calculations done in\n# this block, so we write it to c\nif cacheIndex == 0\n    c[blockIdx().x] = cache[1]\nend\n\nreturn nothing\nendAnd that\'s it! We have made it through the kernel. Now all we have to do is run the kernel on a GPU:function main()\n\n    # Initialise variables\n    N::Int64 = 33 * 1024\n    threadsPerBlock::Int64 = 256\n    blocksPerGrid::Int64 = min(32, (N + threadsPerBlock - 1) / threadsPerBlock)\n\n    # Create a,b and c\n    a = CuArrays.CuArray(fill(0, N))\n    b = CuArrays.CuArray(fill(0, N))\n    c = CuArrays.CuArray(fill(0, blocksPerGrid))\n\n    # Fill a and b\n    for i in 1:N\n        a[i] = i\n        b[i] = 2*i\n    end\n\n    # Execute the kernel. Note the shmem argument - this is necessary to allocate\n    # space for the cache we allocate on the gpu with @cuDynamicSharedMem\n    @cuda blocks = blocksPerGrid threads = threadsPerBlock shmem =\n    (threadsPerBlock * sizeof(Int64)) dot(a,b,c, N, threadsPerBlock, blocksPerGrid)\n\n    # Copy c back from the gpu (device) to the host\n    c = Array(c)\n\n    local result = 0\n\n    # Sum the values in c\n    for i in 1:blocksPerGrid\n        result += c[i]\n    end\n\n    # Check whether output is correct\n    println(\"Does GPU value \", result, \" = \", 2 * sum_squares(N - 1))\nend\n\nmain()\nmain() starts by initialising several variables, including N which sets the size of a, b and c. We also initialise the number of threads we want the GPU to use per block and the number of blocks we want to use on the GPU. Next, we use CuArrays to create a, b and c and to fill a and b. Then, we use @cuda to execute the kernel on the GPU:@cuda blocks = blocksPerGrid threads = threadsPerBlock shmem =\n(threadsPerBlock * sizeof(Int64)) dot(a,b,c, N, threadsPerBlock, blocksPerGrid)Note that in addition to setting the number of blocks and threads we want the GPU to use, we set a value for shmem. shmem describes the amount of dynamic shared memory we need to allocate for the kernel - see below for more details. Since we use @cuDynamicSharedMem to make an array of size threadsPerBlock full of Int64s in the kernel, we need to allocate (threadsPerBlock * sizeof(Int64) bytes of space in advance when we call @cuda.After executing the kernel on GPU, we copy c back to the host (CPU). At this point, c is an array whose length equals the number of blocks in the grid. Each element in c is equal to the sum of the values calculated by the threads in a block. We need to sum the values of c to find the final result of the vector dot product:# Sum the values in c\nfor i in 1:blocksPerGrid\n    result += c[i]\nendFinally, we do a sanity check to make sure the output is correct. For completeness, this is the function sum_squares():function sum_squares(x)\n    return (x * (x + 1) * (2 * x + 1) / 6)\nendAnd that is it! We now have a complete Julia script which calculates a vector dot product on a GPU, making use of shared memory and synchronisation. In the next section, we will discuss streaming."
},

{
    "location": "Vector_dot_product/#A-Note-on-Static-and-Dynamic-Allocation-1",
    "page": "Shared Memory and Synchronisation",
    "title": "A Note on Static and Dynamic Allocation",
    "category": "section",
    "text": "In the first line of the kernel, we call @cuDynamicSharedMem. @cuDynamicSharedMem has a sister function, @cuStaticSharedMem. Like @cuDynamicSharedMem, @cuStaticSharedMem allocates arrays in shared memory. However unlike @cuDynamicSharedMem, @cuStaticSharedMem allocates arrays statically rather than dynamically. Memory that is statically allocated is allocated at compilation time, whereas memory that is dynamically allocated is allocated at program execution. We used @cuDynamicSharedMem in our example because one of the command line arguments for @cuDynamicSharedMem was a kernel command line argument (threadsPerBlock). Because the value of the kernel command line argument is not known at compilation time, dynamic rather than static memory allocation was required.A consequence of using dynamic rather than static memory allocation was that we had to specify how much memory @cuDynamicSharedMem would need in our @cuda call. Otherwise, there is no way @cuda could know the correct amount of shared memory to allocate in advance, since @cuDynamicSharedMem does not determine how much shared memory it will need until it runs."
},

{
    "location": "Streaming/#",
    "page": "Streaming",
    "title": "Streaming",
    "category": "page",
    "text": ""
},

{
    "location": "Streaming/#Streaming-1",
    "page": "Streaming",
    "title": "Streaming",
    "category": "section",
    "text": "In CUDA, a stream is a sequence of operations executed in order on a device. We can use multiple streams to execute multiple sequences of operations sequentially. A common reason for using multiple streams in GPU programming is to \'hide\' the time taken for data transfer. Often copying data between the host and the device is one of the slowest steps in a GPU program. By writing your program with streams, you can split your data into chunks and have the GPU analysing a chunk in one stream whilst simultaneously copying a chunk of data to the GPU in another stream.It should be noted that Julia\'s support for streaming in GPU programming is still rudimentary. As you will see, it is easy to stream kernel execution (analysis), but the ideal of streaming both data transfer and analysis is more challenging. Streaming both data transfer and analysis will require us to write much lower level code than is usually seen in Julia. Let\'s start with the simpler task of streaming our analysis."
},

{
    "location": "Streaming/#Streaming-our-Analysis-1",
    "page": "Streaming",
    "title": "Streaming our Analysis",
    "category": "section",
    "text": "In this example, we will return to our favourite problem of vector addition. A script which will carry out vector addition in two streams is shown below:using CuArrays, CUDAnative, CUDAdrv\nusing Test\n\n# Kernel\nfunction add!(a,b,c, index)\n    c[index] = a[index] + b[index]\n    return nothing\nend\n\nfunction main()\n\n    # Initialise a, b and c\n    a = CuArrays.CuArray(fill(0, 10))\n    b = CuArrays.CuArray(fill(0, 10))\n    c = CuArrays.CuArray(fill(0, 10))\n\n    # Put values in a and b\n    for i in 1:10\n        a[i] = -i\n        b[i] = i * i\n    end\n\n    # Create two streams\n    s1 = CuStream()\n    s2 = CuStream()\n\n    # Call add! asynchronously in two streams\n    for i in 1:2:min(length(a), length(b), length(c))\n        @cuda threads = 1 stream = s1 add!(a,b,c, i)\n        @cuda threads = 1 stream = s2 add!(a,b,c, i+1)\n    end\n\n    # Copy arrays back to host (CPU)\n    a=Array(a)\n    b=Array(b)\n    c=Array(c)\n\n    # Check the addition worked\n    for i in 1:length(a)\n        @test a[i] + b[i] ≈ c[i]\n    end\n\nend\n\nmain()So what have we changed to introduce streaming to the classic vector addition example? Firstly, the kernel is slightly different:# Kernel\nfunction add!(a,b,c, index)\n    c[index] = a[index] + b[index]\n    return nothing\nendWe now pass the index into the vectors as an argument, rather than relying on the thread or block index.As previously, main() begins by creating a, b and c and putting values in a and b. The next change is halfway through main(), where we create two streams:# Create two streams\ns1 = CuStream()\ns2 = CuStream()This is fairly self explanatory - CuStream() is a function from CUDAnative which creates a stream. Next, we use the two streams to execute the kernel simultaneously in multiple streams:# Call add! asynchronously in two streams\nfor i in 1:2:min(length(a), length(b), length(c))\n    @cuda threads = 1 stream = s1 add!(a,b,c, i)\n    @cuda threads = 1 stream = s2 add!(a,b,c, i + 1)\nendLet\'s walk through what this for loop is doing. This for loop iterates over the length of the vectors in steps of size 2. In each iteration, we execute the kernel once in the first stream (s1) using i as an index for vector addition, and once in the second stream (s2) using i + 1 as the index for vector addition. Thus, in each interation of the for loop the values of c[i] and c[i + 1] are calculated in seperate streams. main() then finishes as before by copying a, b and c back to the host and checking that the results of the calculation are correct.Hopefully you agree that executing the kernel across two streams was extremely easy in this example. However, it was also fairly pointless. We basically wrote some extra code to execute add! on two elements of c at a time, whereas our previous example without streaming calculated every element of c in parallel using threads.Streaming can really come in to its own when used to stagger data transfer and analysis between streams. This is possible in Julia for GPU applications, but comes with a health warning..."
},

{
    "location": "Streaming/#Health-Warning:-Low-Level-Code-Alert-1",
    "page": "Streaming",
    "title": "Health Warning: Low Level Code Alert",
    "category": "section",
    "text": "To stagger data transfer and analysis between streams, we will not be able to use CUDAnative or CUarrays. This is because CuArrays.CuArray(), the function we have used to copy data from host to device up to now, executes synchronously. This means that the function does not return until the data transfer is complete. If we try to stream data transfer with a synchronous data transfer function, we will not acheive any speed up because no other streams can receive and start executing any further instructions until the current stream\'s data transfer is complete and the function has returned.To acheive speed-up by streaming data transfer, the process of data transfer must be asynchronous. If our data transfer function is asynchronous, it can return before finishing data transfer and we can submit other instructions to other streams whilst data transfer is still ongoing in the first stream. This can be a powerful way to speed up programs. It can also be useful if our data is too large to load on to the GPU all at once, as now we can upload and analyse our data in chunks whilst minimising the time the GPU spends sitting idle waiting for data to finish copying.Unlike CuArrays and CUDAnative, CUDAdrv provides support for asynchronous data transfer. The code in the next example is very analogous to CUDA C or C++ code and the example below does in fact include a (very short) CUDA C script. In practice, this means we will have to think about memory management more than previously and our code will be less pretty. If you have never written code in a lower level language like C, you may struggle to follow the next section. If you are struggling, do not panic and just move on to the next section. In practice, you can often speed up your code a lot by porting to GPU without using streams."
},

{
    "location": "Streaming/#Streaming-Data-Transfer-and-Analysis-1",
    "page": "Streaming",
    "title": "Streaming Data Transfer and Analysis",
    "category": "section",
    "text": "Let\'s rewrite our vector addition example so that it uses streaming for both data transfer and analysis.using CUDAdrv, Test\n\n# \'Turn on\' device\ndev = CuDevice(0)\nctx = CuContext(dev)\n\n# Read in C code\nmd = CuModuleFile(joinpath(@__DIR__, \"vadd.ptx\"))\nvadd = CuFunction(md, \"kernel_vadd\")\n\n# Make data\ndims = 100\na = round.(rand(Float32, dims) * 100)\nb = round.(rand(Float32, dims) * 100)\nc = similar(a)\n\n# Allocate memory for a and b on device stream 1\nbuf_a1 = Mem.alloc(sizeof(Float32))\nbuf_b1 = Mem.alloc(sizeof(Float32))\n\n# Allocate memory for a and b on device stream 2\nbuf_a2 = Mem.alloc(sizeof(Float32))\nbuf_b2 = Mem.alloc(sizeof(Float32))\n\n# Allocate memory for c on device\nd_c1 = Mem.alloc(sizeof(Float32))\nd_c2 = Mem.alloc(sizeof(Float32))\n\n# Make streams\ns1 = CuStream()\ns2 = CuStream()\n\n# Iterate over arrays in increments of 2\nfor i in 1:2:dims\n\n    # Asynchronously copy a[i] and a[i+1] onto device\n    Mem.upload!(buf_a1, Ref(a, i), sizeof(Float32), s1, async = true)\n    Mem.upload!(buf_a2, Ref(a, i+1), sizeof(Float32), s2, async = true)\n\n    # Asynchronously copy b[i] and b[i+1] onto device\n    Mem.upload!(buf_b1, Ref(b, i), sizeof(Float32), s1, async = true)\n    Mem.upload!(buf_b2, Ref(b, i+1), sizeof(Float32), s2, async = true)\n\n    # Call vadd to run on gpu\n    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a1, buf_b1, d_c1,\n    threads = 1, stream = s1)\n    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a2, buf_b2, d_c2,\n    threads = 1, stream = s2)\n\n    # Asynchronously copy c[i] and c[i+1] back to host\n    Mem.download!(Ref(c, i), d_c1, sizeof(Float32), s1, async = true)\n    Mem.download!(Ref(c, i+1), d_c2, sizeof(Float32), s2, async = true)\n\n\nend\n\n# Check it worked\n@test a+b ≈ c\n\n# Destroy context\ndestroy!(ctx)This is probably looking pretty alien at this point - don\'t worry, we will work through it slowly. Starting from the beginning:using CUDAdrv, Test\n\n# \'Turn on\' device\ndev = CuDevice(0)\nctx = CuContext(dev)The script starts as usual by loading some Julia packages. Then there are two lines we haven\'t seen before calling CuDevice and CuContext. These two lines essentially work together to create a context on the GPU. A GPU context can be thought of as analogous to a CPU process, so by running these two lines we are creating an address space and allocated resources on the GPU where our kernel can run and we can copy data. Once we have finished working on the GPU, we will need to destroy our context so the system can clean up the resources allocated there. Both creating the context and destroying it are taken care of behind the scenes by CUDAnative, which is why we have been able to ignore these functions until now.The lines of code that follow context creation will either delight or alarm you, depending on your relationship with C:# Read in C code\nmd = CuModuleFile(joinpath(@__DIR__, \"vadd.ptx\"))\nvadd = CuFunction(md, \"kernel_vadd\")So what is going on here? Well, first CuModuleFile reads in a file called vadd.ptx. In the next line, we use CuFunction to make a handle to a function called kernel_vadd defined in vadd.ptx, then assign that function handle to vadd. The outcome of these two lines is that a function called kernel_vadd defined in vadd.ptx becomes callable in Julia, and can be called using vadd().So what do the contents of vadd.ptx look like?//\n// Generated by NVIDIA NVVM Compiler\n//\n// Compiler Build ID: CL-23083092\n// Cuda compilation tools, release 9.1, V9.1.85\n// Based on LLVM 3.4svn\n//\n\n.version 6.1\n.target sm_30\n.address_size 64\n\n	// .globl	kernel_vadd\n\n.visible .entry kernel_vadd(\n	.param .u64 kernel_vadd_param_0,\n	.param .u64 kernel_vadd_param_1,\n	.param .u64 kernel_vadd_param_2\n)\n{\n	.reg .f32 	%f<4>;\n	.reg .b32 	%r<5>;\n	.reg .b64 	%rd<11>;\n\n\n	ld.param.u64 	%rd1, [kernel_vadd_param_0];\n	ld.param.u64 	%rd2, [kernel_vadd_param_1];\n	ld.param.u64 	%rd3, [kernel_vadd_param_2];\n	cvta.to.global.u64 	%rd4, %rd3;\n	cvta.to.global.u64 	%rd5, %rd2;\n	cvta.to.global.u64 	%rd6, %rd1;\n	mov.u32 	%r1, %ctaid.x;\n	mov.u32 	%r2, %ntid.x;\n	mov.u32 	%r3, %tid.x;\n	mad.lo.s32 	%r4, %r2, %r1, %r3;\n	mul.wide.s32 	%rd7, %r4, 4;\n	add.s64 	%rd8, %rd6, %rd7;\n	ld.global.f32 	%f1, [%rd8];\n	add.s64 	%rd9, %rd5, %rd7;\n	ld.global.f32 	%f2, [%rd9];\n	add.f32 	%f3, %f1, %f2;\n	add.s64 	%rd10, %rd4, %rd7;\n	st.global.f32 	[%rd10], %f3;\n	ret;\n}Above are the contents of vadd.ptx - unless you are an assembly code expert I suspect you will agree this is not very informative. PTX stands for Parallel Thread eXecution and we will briefly discuss two ways to make PTX code in this tutorial. The first way to make PTX code is by compiling CUDA C code. So what did the CUDA C code that made this PTX look like?extern \"C\" {\n\n__global__ void kernel_vadd(const float *a, const float *b, float *c)\n{\n    int i = blockIdx.x *blockDim.x + threadIdx.x;\n    c[i] = a[i] + b[i];\n}\n\n}Even if you don\'t know C, hopefully you will be able to see that this function calculates a unique thread index (i) based on the block the thread is in and the thread\'s index within that block, then adds the values of a and b at that index and stores them in c. This is very similar to many of the kernels we have written before. We can make PTX code from CUDA C by executing the following command in a terminalnvcc --ptx /path/to/cuda_C_file.cuIt is also possible to make PTX code from Julia code using functions in CUDAnative such as code_ptx() - see http://juliagpu.github.io/CUDAnative.jl/latest/lib/reflection.html# CUDAnative.code_ptx for details.Ok, so the outcome of all of the above is that we can now have a calleable GPU compatible function called vadd which performs vector addition. Next, we create our data:# Make data\ndims = 100\na = round.(rand(Float32, dims) * 100)\nb = round.(rand(Float32, dims) * 100)\nc = similar(a)This is done on a CPU and there is nothing really to note except that we are not using CuArrays. Next, we need to allocate memory on the GPU for a, b and c:# Allocate memory for a and b on device stream 1\nbuf_a1 = Mem.alloc(sizeof(Float32))\nbuf_b1 = Mem.alloc(sizeof(Float32))Again, this is the first time we have had to do this because previously it was taken care of for us by CUDAnative. In our streaming strategy, we are going to copy one element of a and b to each stream, calculate the value of the corresponding value of c, store that result in c, then copy the value of c back to the host. This means we need to allocate space for one element of a, b and c for each stream. There are two streams, so in practice this means we need to allocate space twice. In the code above, we are allocating space for one element of a and one element of b for the first stream. Each element in a and b is a Float32, so we use sizeof() to work out how many bytes we need to allocate. We also need to allocate space for a and b in stream 2, so we write:# Allocate memory for a and b on device stream 2\nbuf_a2 = Mem.alloc(sizeof(Float32))\nbuf_b2 = Mem.alloc(sizeof(Float32))Next, we allocate space for c on both streams and create the streams:# Allocate memory for c on device\nd_c1 = Mem.alloc(sizeof(Float32))\nd_c2 = Mem.alloc(sizeof(Float32))\n\n# Make streams\ns1 = CuStream()\ns2 = CuStream()Now we are ready to actually start streaming data transfer and analysis. Like in the \'analysis only\' streaming example, we do this here by iterating over our arrays in increments of size 2.# Iterate over arrays in increments of 2\nfor i in 1:2:dims\n\n    # Asynchronously copy a[i] and a[i+1] onto device\n    Mem.upload!(buf_a1, Ref(a, i), sizeof(Float32), s1, async = true)\n    Mem.upload!(buf_a2, Ref(a, i+1), sizeof(Float32), s2, async = true)\n\n    # Asynchronously copy b[i] and b[i+1] onto device\n    Mem.upload!(buf_b1, Ref(b, i), sizeof(Float32), s1, async = true)\n    Mem.upload!(buf_b2, Ref(b, i+1), sizeof(Float32), s2, async = true)\n\n    # Call vadd to run on gpu\n    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a1, buf_b1, d_c1,\n    threads = 1, stream = s1)\n    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), buf_a2, buf_b2, d_c2,\n    threads = 1, stream = s2)\n\n    # Asynchronously copy c[i] and c[i+1] back to host\n    Mem.download!(Ref(c, i), d_c1, sizeof(Float32), s1, async = true)\n    Mem.download!(Ref(c, i+1), d_c2, sizeof(Float32), s2, async = true)\n\n\nendAs before, we operate on the ith element in one stream and the i + 1th element in the other stream. In each stream, the order of operations is to copy (Mem.upload!) the value of the ith or i + 1th element of a and b from the host to the pre-allocated space on the device. Then we run the kernel (using cudacall) and copy the result back to the host (Mem.download!).There are several things worth noting here. First, we must explicitly specify async = true to Mem.upload! and Mem.download! if we want them to execute asynchronously - as a default these functions are synchronous. Second, note that we call the kernel using cudacall rather than @cuda - this is because @cuda is part of CUDAnative which does not support asynchronous copies. Thirdly, note that we must specify which stream we are using for each of these functions - otherwise all of these functions will execute on the same stream as a default.Finally, we finish our script by checking the results of our calculation and destroying the device context:# Check it worked\n@test a+b ≈ c\n\n# Destroy context\ndestroy!(ctx)This example demonstrates that it is possible to stream data transfer and analysis in Julia. However, it should be noted that the code we wrote was so low level that almost every line we wrote was directly analogous to a CUDA C command. I would argue that the benefit of writing a program such as this in Julia versus C is debateable. There might be situations where it makes sense to write this type of program in Julia (for example, if you already had a huge Julia code base, a small part of which you wanted to port to GPU), but often it might actually be easier to use C.In the next section, we will consider some Julia specific aspects of writing GPU compatible software."
},

{
    "location": "Streaming/#References-1",
    "page": "Streaming",
    "title": "References",
    "category": "section",
    "text": "The data streaming and analysis example above was based on code taken from here: https://github.com/JuliaGPU/CUDAdrv.jl/tree/master/examples"
},

{
    "location": "Challenges/#",
    "page": "Challenges in Julia GPU Software Development",
    "title": "Challenges in Julia GPU Software Development",
    "category": "page",
    "text": ""
},

{
    "location": "Challenges/#Challenges-in-Julia-GPU-Software-Development-1",
    "page": "Challenges in Julia GPU Software Development",
    "title": "Challenges in Julia GPU Software Development",
    "category": "section",
    "text": "Hopefully I have managed to convince you that developing GPU software with Julia can be a useful thing to do. This section outlines some common challenges and ‘gotchas!’ you might encounter when developing GPU software in Julia."
},

{
    "location": "Challenges/#isbitstype-1",
    "page": "Challenges in Julia GPU Software Development",
    "title": "isbitstype",
    "category": "section",
    "text": "Julia makes a distinction between “bits” values and heap allocated “boxed” values. Bits values are stored inline in memory, whereas boxed values contain pointers to objects allocated elsewhere. @cuda only supports the passing of bits values as kernel arguments. This is because bits values are of a fixed, determinate size, so @cuda is able to allocate the correct amount of memory for these values on the device. In contrast, @cuda can’t determine the size of a value with pointers, and so @cuda will throw an error. You can determine whether your values are bits values using the function isbits(), which will return true if your value is bits and false otherwise. Something to be aware of is that strings are not bits values – if you want to work with strings on GPUs you will need to convert your string to an array of chars."
},

{
    "location": "Challenges/#Impenetrable-Error-Messages-1",
    "page": "Challenges in Julia GPU Software Development",
    "title": "Impenetrable Error Messages",
    "category": "section",
    "text": "When developing GPU software, you will inevitably write kernels with bugs in them. If a bug in your kernel causes the kernel to exit and return an error message, you will often see error messages such as these:ERROR: LLVM IR generated for Kernel(CuDeviceArray{Int64,1,CUDAnative.AS.Global}) is not GPU compatibleOrThe error about not returning nothingThe first error message broadly means that you tried to do something which could not compile to GPU code. Common reasons this might happen include:You tried to use a Julia feature which is not supported by CUDAnative (see below).\nYour kernel contains unexpected dynamic behavior.\nYour kernel contains type instabilities.The second error message indicates that your code compiled successfully (unless it is preceded by the first error message) but that something went wrong during runtime, hence it returned something other than nothing.Unfortunately, you will never see the sort of error message you are probably accustomed to in Julia, where a description of why the code failed to compile or run and a line number are provided. A description of why this is and macros you can use to help debug are described here: http://juliagpu.github.io/CUDAnative.jl/latest/man/troubleshooting.html. In practice, I often debug by iteratively commenting out half of the remaining lines of code in my kernel to identify which line(s) are causing problems."
},

{
    "location": "Challenges/#CUDAnative-Does-Not-Support-All-of-Julia-1",
    "page": "Challenges in Julia GPU Software Development",
    "title": "CUDAnative Does Not Support All of Julia",
    "category": "section",
    "text": "This is actually unsurprising when you think about it – Julia is pretty big and CUDAnative is maintained by a relatively small number of people. Plus, actually working out how you would implement support for some of Julia’s features on a GPU is really not trivial.Unfortunately, the subset of Julia supported by CUDAnative is undocumented. Some commonly used Julila features not supported by CUDAnative at the time of writing include:Strings\nType conversion\nRecursion\nExceptions\nCalls to the Julia runtime\nGarbage collection"
},

{
    "location": "Profiling/#",
    "page": "Profiling",
    "title": "Profiling",
    "category": "page",
    "text": ""
},

{
    "location": "Profiling/#Profiling-1",
    "page": "Profiling",
    "title": "Profiling",
    "category": "section",
    "text": ""
},

{
    "location": "Performance_thoughts/#",
    "page": "Thoughts on Performance",
    "title": "Thoughts on Performance",
    "category": "page",
    "text": ""
},

{
    "location": "Performance_thoughts/#Thoughts-on-Performance-1",
    "page": "Thoughts on Performance",
    "title": "Thoughts on Performance",
    "category": "section",
    "text": "Talk about warps and branching here. Add a \'Things to consider/Gotchas\' page where I discuss isbits?"
},

{
    "location": "Summary/#",
    "page": "Summary",
    "title": "Summary",
    "category": "page",
    "text": ""
},

{
    "location": "Summary/#Summary-1",
    "page": "Summary",
    "title": "Summary",
    "category": "section",
    "text": "Table of GPU Julia resources and what they can do?"
},

{
    "location": "Further_Reading/#",
    "page": "Further Reading",
    "title": "Further Reading",
    "category": "page",
    "text": ""
},

{
    "location": "Further_Reading/#Further-Reading-1",
    "page": "Further Reading",
    "title": "Further Reading",
    "category": "section",
    "text": "CUDA by Example Githubs for packages"
},

{
    "location": "About_the_author/#",
    "page": "About the Author",
    "title": "About the Author",
    "category": "page",
    "text": ""
},

{
    "location": "About_the_author/#Jenni-Westoby-1",
    "page": "About the Author",
    "title": "Jenni Westoby",
    "category": "section",
    "text": "I am a PhD student studying bioinformatics (computational biology) at the University of Cambridge and the Wellcome Trust Sanger Institute. I got interested in GPU software development because of its potential to accelerate bioinformatics software. I hope you enjoyed reading this tutorial and that you found it useful :)"
},

]}
