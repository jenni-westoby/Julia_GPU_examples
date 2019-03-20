#  Accessing GPUs

You basically have three options for accessing GPUs:

1. Access GPUs on a commercial cloud based platform e.g. [AWS](https://aws.amazon.com/).
2. Access GPUs using a HPC you have access to through your company/academic institution.
3. Use the GPU in your own computer

# Option 1

If you are going for option 1, the cloud based platform you are using should have instructions for how to gain access to the GPUs you are paying for. It is quite likely that the cloud based platform will already have the CUDA toolkit installed, so once you have managed to log in you are probably good to go!

# Option 2

If you are going for option 2, talk to the HPC administrator to find out how to get access to the GPUs and check that the CUDA toolkit is already installed.

# Option 3

If you are going for option 3, things are about to get interesting.

These are some instructions you can try following to get the GPU in your computer working in a way that will allow Julia to interact with it. However, I have not carried out extensive testing on different hardware platforms, and thus I make no promises that they will work and I will not offer support if they fail. These instructions assume you are working in a Linux environment. I do not know how to make your GPU work in a Windows or Mac environment.

**Step 1:** Check that your computer has a GPU.

**Step 2:** Check that your computer has a GPU that is actually supported by CUDA (CUDA is only supported on GPUs made by NVIDIA, and is not supported on older NVIDIA GPUs).

**Step 3:** Replace your operating system with a fresh install of Ubuntu 18.04. No, I am not joking. Remember to back up your files before you install a new operating system just in case something goes wrong.

**Step 4:** Open a terminal and type in the following commands:

```
sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt update

sudo ubuntu-drivers autoinstall

```

**Step 5:** Reboot your computer

**Step 6:** Open a terminal and type in the following:

```
sudo apt install nvidia-cuda-toolkit gcc-6

nvcc --version

```

If your install has worked, the final command should print out some version information.

**Step 7:** Assuming all of the above worked, install the latest version of [Julia](https://julialang.org/downloads/) and continue working through this tutorial.

**Disclaimer** The above are NOT authoritative instructions on how to install the CUDA toolkit, they are instructions that worked for me. You should visit the [NVIDIA website](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for authoritative instructions.

**Disclaimer** I am writing these instructions in spring 2019, pre-Brexit. If you are reading this disclaimer and the year is 2021 or greater, these instructions may be woefully out of date and better ignored. I would also be grateful if you travel back in time and fix parliament.

# Surely I don't really need to install a new operating system?

The CUDA toolkit and drivers are notoriously difficult to install. It might sound a bit mad, but in my experience installing a fresh version of Ubuntu 18.04 is the most reliable method for installing CUDA. Note, steps 4-6 are taken from the second answer [here](https://askubuntu.com/questions/1028830/how-do-i-install-cuda-on-ubuntu-18-04).
