# Warps and Branching

Based on your reading of this tutorial so far, you could be forgiven for thinking that all threads execute entirely independently of each other. In fact, this is not the case. In CUDA architecture, threads are split into groups of 32. Each group of 32 threads executes in 'lockstep' with each other. So in other words, within a warp each thread is executing the same line of your program but with different data. A group of 32 threads that execute in lockstep is referred to as a 'warp'.

Warps can have some important performance implications. For example, consider the pseudocode if/else statement below:

```
if condition1
  do thing1
else
  do thing2
end
```

Because all of the threads in a warp execute in lockstep, if ```condition1``` is only true for even just one thread in the warp, all of the threads in that warp must wait until that thread has executed ```do thing1```. Then, the thread for which ```condition1``` was true for must wait for all of the other threads to execute the ```else``` condition and ```do thing2```. Finally, all the threads can move on to the next line of code beyond the if/else statement.

This if/else statement is an example of 'branching' code and contains two branches - a branch corresponding to ```thing1``` and a branch for ```thing2```.This example illustrates that branching can potentially substantially slow down your code, as all of the threads in our warp had to wait for execution of both branches to complete. This is in contrast to CPU threads, where each thread would only have to execute the branch corresponding to whether ```condition1``` was true or false for that thread. The take home message from this section is to avoid heavily branching GPU code where possible, as it is possible each warp will have to wait for every branch to be executed, dramatically slowing down your program.
