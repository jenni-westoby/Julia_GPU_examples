
<a id='Some-Background-on-GPUs-1'></a>

# Some Background on GPUs


This section gives a very light overview of what GPUs are and some relevant information on how they are organised for developers.


<a id='What-are-GPUs?-1'></a>

# What are GPUs?


You have probably heard of a type of computer chip called a CPU. GPUs are a different type of computer chip. The major difference between CPUs and GPUs is how each chip is organised. Whereas CPUs tend to have a small number of cores and threads, GPUs can have orders of magnitudes more. On GPUs, you can write code that parallelises across thousands of threads, something that simply would not be possible on CPUs.


<a id='How-are-GPUs-organised?-1'></a>

# How are GPUs organised?


![](images/grid_threads_blocks.png)


<a id='Host-and-Device-1'></a>

# Host and Device


Introduce kernels here


<a id='Glossary-of-terms-1'></a>

# Glossary of terms


**Grid**


**Block**


**Thread**


**Host**


**Device**


**Kernel**


Hopefully this has got you hyped to get started developing on GPUs! Next we will make sure we have access to a usable GPU.

