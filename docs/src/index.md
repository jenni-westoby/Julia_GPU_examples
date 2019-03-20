# Learn to Develop GPU Software with Julia

A number of packages have been developed for [Julia](https://julialang.org/) that enable developers to write code that will run on graphics processing units ([GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit)). But until now, open source documentation which would enable a developer to go from zero [CUDA](https://developer.nvidia.com/cuda-zone) knowledge to making effective use of these packages has not existed. This tutorial aims to change that. I will guide you through the full workflow of writing a basic program in Julia that will run on a GPU. Things you will learn from this tutorial include:

- What a GPU is and why writing code to run on GPUs might be useful.
- How to set up your computer and install the graphics drivers necessary to make use of your GPU.
- How to write Julia code that will run on a GPU in a parallel manner using threads, blocks, shared memory and streaming.
- How to benchmark Julia code that runs on GPUs.

This tutorial assumes you already know how to write code in Julia. Julia is a high level, syntactically intuitive language, so if you do not know Julia but do know other programming languages you may be able to follow along anyway. To get started coding in Julia, I recommend reading some of the [Julia Documentation](https://docs.julialang.org/en/v1.0/).

This tutorial is intended to provide a foundation for developers to start writing Julia code that runs on GPUs. It is not intended to be a comprehensive guide to everything you could conceivably do on a GPU using Julia. Hopefully, you will find it instead gives a gentle introduction to some core concepts in GPU programming and how you can use them in Julia. I focus on the CUDAdrv, CUDAnative and CuArrays packages in this tutorial, although other Julia packages for using GPUs exist - see the [JuliaGPU github page](https://github.com/JuliaGPU) for details.
