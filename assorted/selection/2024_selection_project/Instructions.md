# Wits HPC Selection Project

Over the last few weeks, we have taught you the basics of HPC. We now want you to put your knowledge to the test by running a compute-intensive task, Bitonic Sort, across a 3-node virtual cluster using OpenHPC.

# Bitonic Sort

The bitonic sort algorithm is based on the concept of a sorting network. It efficiently sorts a given list or array of elements in ascending or descending order. The source code can be found at: <wits github>. Credits to Sayfullah Jumoorty and Muhammed Muaaz Dawood.

### Compiling bitonic_sort

```
cd bitonic_sort/src
```

You can make changes to the Makefile:

```
OMPFLAG = 
CC = g++
MPI_CC = 
CFLAGS =
```

Create your executable:

```
make
```

There should be a bitonic executable in your directory.

### Running bitonic_sort

You first need to generate your inputs. Do not change the n value in gen.sh.

```
cd bitonic_sort/gen

bash gen.sh
```

Now you can run your program:

```
cd bitonic_sort/src
	
./bitonic #or mpirun ./bitonic
```

# 3-node OpenHPC cluster

This course provides a complete guide on how to set up the virtual 3-node cluster: https://events.chpc.ac.za/event/136/

OpenHPC is an open-source project that provides a comprehensive and pre-configured software stack for high-performance computing (HPC) environments. The course presents a step-by-step guide towards the deployment of a virtual HPC cluster using the community-driven open-source HPC software suite - OpenHPC.

We recommend doing this on your personal computer as it is not easily transferable.

# Tasks:

- Submit your scripts to download, install and run Bitonic Sort.
- Submit your Makefile.
- Submit your run.log.
- Submit screenshot of your 3 nodes running using ```vagrant status```.