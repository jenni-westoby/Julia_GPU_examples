# Further Reading

I hope you found this tutorial informative and that you now feel ready to write your own GPU software in Julia. Here are some additional resources that you might find useful on your GPU journey:

- [**CUDA Documentation**](https://juliagpu.gitlab.io/CUDA.jl/)
- [**JuliaGPU github**](https://github.com/JuliaGPU). We covered the CUDAdrv, CUDAnative and CuArrays packages in this tutorial, but there are lots of other Julia packages enabling GPU software development out there. This github profile contains a list of Julia GPU package repositories.
- **'CUDA by Example, An Introduction to General-Purpose GPU Programming' by Jason Sanders and Edward Kandrot**. This textbook teaches CUDA C as oppose to Julia, but as CUDAnative is closely based on CUDA C you may still find this a useful read. Good for learning the underlying theory and concepts.
- [**NVIDIA's documentation**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html). Again, this is for CUDA C, but many of the functions in CUDAnative and CUDAdrv are analogous to CUDA C functions.
