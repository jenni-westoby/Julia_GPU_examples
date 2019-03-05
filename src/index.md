#Learn to Develop GPU Software Using Julia

A number of packages have been developed for Julia that enable developers to write Julia code that will run on GPUs. But until now, open source documentation which would enable a developer to go from zero CUDA knowledge to making effective use of these packages has not existed. This tutorial aims to change that. In this tutorial, I will guide you through the full workflow of writing a basic program in Julia that will run on GPU. Things you will learn from this tutorial include:

- How to set up your computer and install the graphics drivers necessary to make use of your GPU.
- What a GPU is and how it works
- How to write Julia code that will run on a GPU in a parallel manner using threads, blocks, shared memory and streaming.
- How to benchmark Julia code that runs on GPUs.

This tutorial focuses on the CUDAdrv, CUDAnative and CuArrays packages, although other Julia packages for using GPUs exist - see https://github.com/JuliaGPU for details.

## Getting Started

```@contents
Pages = ["tutorial/Accessing_GPUs.md"]
Depth = 2
```
