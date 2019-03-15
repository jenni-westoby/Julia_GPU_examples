# Set-up Julia

This section assumes that you have already carried out the instructions in 'Accessing GPUs' and that you now have access to a machine which

1. Contains a CUDA-compatible GPU, and
2. Has a working CUDA toolkit installation.

If this is not the case, many of the below steps will not work because they depend on the two conditions above being true.

Assuming you did follow the instructions in "Accessing GPUs" and you have access to a computer with a GPU and a working CUDA toolkit installation, carry out the steps below to get a Julia environment that you can run this tutorial in.

**Step 1:** Follow the link instructions to install Julia 1.0.3 if you haven't already done so: https://julialang.org/downloads/. I recommend against using your OS's package manager to install Julia, you are unlikely to get the correct version.

**Step 2** Within Julia, install CUDAdrv, CUDAnative and CuArrays using the commands below:

```
]
add CUDAdrv
add CUDAnative
add CuArrays
```

Note - this step is likely to fail if you do not have a working CUDA installation.

# Common Problems

**I'm sure I installed the CUDA toolkit correctly yesterday but now nothing is working!**

Try rebooting your machine. Sometimes when your machine goes to sleep/ you close your laptop lid/ whatever, the graphics driver configuration is slightly different when it wakes up again. Rebooting your machine fixes this problem.

**I installed the packages in a slightly different order to you and they didn't install properly.**

Try uninstalling the packages and re-installing them in the same order as me.

**It turns out there is something wrong with my CUDA installation.**

This is a very common problem. You could try following the instructions in "Set up your GPU" to install the CUDA toolkit. If none of the options there work for you then I'm afraid I am out of suggestions :(
