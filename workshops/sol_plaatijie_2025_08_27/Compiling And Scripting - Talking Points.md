# Compiling Software

## Why ?

Why do we want to compile software? We have package managers right? Why don't we just use that?

> Refer back to hardware section

As we know, different CPUs (and GPUs) have different physical architectures and different instruction sets, which the chip designers make available for some sort of standard.

When we use `apt`, `dnf`, `pacman`, etc. to install a package, we get a version of that software that is compiled to work on **ANY** CPU that has the architecture the package manager has detected on your system.

With all this in mind, once again, why is this undesirable in a high performance environment?
- Chips cost money, server chips cost more, we want to squeeze every FLOP out of them that we can, and that means software compiled and optimised to work on the specific CPU on the specific machine/s we are working on
- This gets more complex as we scale:
	- In purely homogeneous computing environments, we can optimise everything down to the smallest detail
	- In heterogeneous environments it becomes a bit more complicated, and we have to find a balance between software optimisation and scalability

## How ?

Unfortunately, every software package can choose from a few different options of compilation methods. Since most high performance software is written in a mixture of `C`, `C++`, and `Fortran`, popular build systems include `Make`, `CMake`, and `autotools`

Thankfully, **most** software comes with a `INSTALL.md` or `README.md` that details exactly what steps to take and what options are available to us, we're going to use `hwloc`, or the Portable Hardware Locality package, as a case study.

# HWLOC

> The Portable Hardware Locality (hwloc) software package provides a **portable abstraction** (across OS, versions, architectures, …) of the **hierarchical topology of modern architectures**, including NUMA memory nodes (DRAM, HBM, non-volatile memory, CXL, etc.), processor packages, shared caches, cores and simultaneous multithreading. It also gathers various system attributes such as cache and memory information as well as the locality of I/O devices such as network interfaces, InfiniBand HCAs or GPUs.

A lot of different packages use `hwloc` for more efficient mapping of processes to hardware (i.e. determining matrix sizes, etc.). Its also used for resource management and job scheduling across small and large clusters, so we can consider it a foundational piece of software in the world of HPC.

## Download

You would generally go to your chosen search engine and type `hwloc` or any other package name and go from there, but here is the link: https://www.open-mpi.org/projects/hwloc/. On the page, you can find a link to the downloads page, and we're going to use `wget` to retrieve the latest stable version of the package

```bash
wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.2.tar.bz2
```

Okay we now have the compressed download, and need to extract it:

```bash
tar -xvfj hwloc-2.12.2.tar.bz2
```

And move into the directory:

```bash
cd hwloc-2.12.2
```

## Compile

And look at that, it has a `README`! Lets delve into that to see what we should do, and to understand the package.

Unfortunately, this is one of the cases where the `README` doesn't give compilation instructions, and the rest isn't obvious. What other options do we have?

We can use the `configure` script! This package is using `autotools`, probably the weirdest/most difficult to identify if you haven't seen it before, but also a very common occurrence. 

To see the options available to us, we type:

> Before running this, check the permissions of the files, why can we run it like this?

```bash
./configure -h
```

After reading through, we have 2 options, depending on if we have `sudo` available to us. If we do, we don't have to set any installation locations differently, because we can install directly to the system. However if we don't, we need to set where to install the program, so that we can use it just in our user space.

> When else would we not want to just install the software into our system?

To do this, we set the `--prefix` option:

```bash
mkdir build
./configure --prefix=$HOME/hwloc-2.12.2/build
```

However I just said that we want optimisations right? So lets also export some compiler optimisation options before we configure:

```bash
export CFLAGS="-O3 -march=native"
export CXXFLAGS="-O3 -march=native"
mkdir build
./configure --prefix=$HOME/hwloc-2.12.2/build
```

These flags are telling our compiler to use the third (and highest) level of optimisation, and to compile the program with the native architecture in mind. There are other optimisation options that you can look up, but these will do for now.

Okay now once the package is configured, you'll see a `Makefile` now exists, so we can now go through the generic steps of using a makefile:

```bash
make -j $(nproc)
```

> What does `-j $(nproc)` ?

```bash
make check
```

```bash
make install
```

We can now check the `build` directory, in which you'll see:

```bash
bin include lib sbin share
```

If we navigate to bin, we'll now be able to use all the tools ourselves!

# Scripting

Okay, now that seems like a lot to do manually on many machines right? What if we could have something do all of the steps for us, and we only have to run a single instruction for each machine?

Turns out we most definitely can, which is where we enter the wonderful world of bash scripts.

To make a bash script, all we need is to create a file with a `.sh` or `.bash` extension and run it using the `bash` command. We could also make it executable and simply run it using that.

So, we're going to delete everything we just did, and redo it by putting it into a script

```bash
rm -r hwloc-2.12
rm hwloc-2.12.2.tar.bz2
```
