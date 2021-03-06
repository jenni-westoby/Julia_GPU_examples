# Challenges in Julia GPU Software Development

Hopefully I have managed to convince you that developing GPU software with Julia can be a useful thing to do. This section outlines some common challenges and ‘gotchas!’ you might encounter when developing GPU software in Julia.

# isbitstype

Julia makes a distinction between __bits__ values and heap allocated __boxed__ values. Bits values are stored inline in memory, whereas boxed values contain pointers to objects allocated elsewhere. ```@cuda``` only supports the passing of bits values as kernel arguments. This is because bits values are of a fixed, determinate size, so ```@cuda``` is able to allocate the correct amount of memory for these values on the device. In contrast, ```@cuda``` cannot determine the size of a value with pointers, and so ```@cuda``` will throw an error. You can determine whether your values are bits values using the function ```isbits()```, which will return ```true``` if your value is bits and false otherwise. Ints, floats, bools and chars are all bits values. Arrays are not bits values, which is which we use ```CuArray()``` in our examples. Something to be aware of is that strings are not bits values – if you want to work with strings on GPUs you will need to convert your string to an array of chars. More complex data structures, such as structs, can be bits values or boxed values depending on whether or not they contain boxed values, such as arrays, strings or pointers.

# Impenetrable Error Messages

When developing GPU software, you will inevitably write kernels with bugs in them. If a bug in your kernel causes the kernel to exit and return an error message, you will often see error messages such as:

```
ERROR: LLVM IR generated for Kernel(CuDeviceArray{Int64,1,CUDAnative.AS.Global}) is not GPU compatible
```
This error message broadly means that you tried to do something which could not compile to GPU code. Common reasons this might happen include:
- You tried to use a Julia feature which is not supported by CUDAnative (see below).
- Your kernel contains unexpected dynamic behavior.
- Your kernel contains type instabilities.


Or

```
KernelError: kernel returns a value of type `Union{Nothing, Int64}`

```

The second error message indicates that your code compiled successfully (unless it is preceded by the first error message, or another compilation error) but that something went wrong during runtime, hence it returned something instead of ```nothing```.

Unfortunately, you will never see the sort of error message you are probably accustomed to in Julia, where a description of why the code failed to compile or run and a line number is provided. A description of why this is and macros you can use to help debug can be found [here](http://juliagpu.github.io/CUDAnative.jl/latest/man/troubleshooting.html). In practice, I often debug the good old fashioned way by iteratively commenting out half of the remaining lines of code in my kernel to identify which line(s) are causing problems.

# CUDAnative Does Not Support All of Julia

This is actually unsurprising when you think about it – Julia is pretty big and CUDAnative is maintained by a relatively small number of people. Plus, actually working out how you would implement support for some of Julia’s features on a GPU is really not trivial.

Unfortunately, the subset of Julia supported by CUDAnative is undocumented. Some commonly used Julila features not supported by CUDAnative at the time of writing include:

- Strings
- Type conversion
- Recursion
- Exceptions
- Calls to the Julia runtime
- Garbage collection
