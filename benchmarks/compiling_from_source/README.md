# Compiling From Source
- [Compiling From Source](#compiling-from-source)
- [What](#what)
- [Why](#why)
- [How](#how)
- [Paths and Environment Variables](#paths-and-environment-variables)
  - [Libraries](#libraries)
- [Optimising](#optimising)
    - [Compiler Flags](#compiler-flags)
    - [Libraries](#libraries-1)
    - [Compilers](#compilers)
- [Summary and Tips](#summary-and-tips)
- [Problems and Issues](#problems-and-issues)
- [Worked Example](#worked-example)
- [Exercises](#exercises)
    - [FFMPEG → Image/Video processing library](#ffmpeg--imagevideo-processing-library)
    - [FFTW → Fourier Transforms](#fftw--fourier-transforms)
  - [Proper HPC Applications](#proper-hpc-applications)
- [Sources](#sources)
- [Resources](#resources)
# What

What is compiling from source? It is one of many different ways of installing software. The other main alternative, which is often used more often is to download a binary (or .exe file in Windows) that you can simply run.

In linux this is often done using the package manager, e.g. `apt-get` in Ubuntu, `yum` for CentOS, etc.

Compiling from source in contrast is basically downloading the source code of a program (often in C/C++ or Fortran) and running the compilation step locally (e.g. `gcc main.cpp -o main`, but just many more files)

# Why

Why would you do such a thing? Isn't it just more work?

There are a few reasons for this:

1. If you are not a super user and you just have a user account, but you want to install software. You cannot then run `apt-get install ...` because of insufficient permissions!
2. Usually binaries that are prebuilt that you can just download are super generic, and don't contain any hardware specific instructions (because it then won't work on any machines that don't support those instructions). While this is general, it is not optimal for performance. If you want the maximal performance, then you would prefer to make use of the special instructions that your hardware provides.
3. In a similar vein, if you want to use different libraries or if you want to make small changes to the source code, then compiling from source is often the only option.

# How

How do we do this?

Often it is done using a combination of the following:

1. Get the code (often `git clone https://github.com/code/code && cd code`)
2. Run `./configure` (to set up and find the libraries)
3. Run `make` to actually run the compilation, which is usually the time consuming step. `make -j` runs it in parallel and speeds it up somewhat.
4. Run `make install` to install the binaries in the correct location.

Often, you'll want to install the software in some other directory than the default one (usually /bin, /usr/bin). To do this, you can usually use the `--prefix` flag with the configure step.

The usage here is something like

`mkdir -p /home/myname/software/gromacs/5/`

Then, replace configure with `./configure --prefix=/home/myname/software/gromacs/5/` and run make as normal

It is also very useful to save the output of the configure and make commands for later perusal, specifically to debug any issues, or to verify that the correct libraries were used.

Another thing that is useful is to have a build script that is reproducible.

- For example, here is a shortened version of one of the build scripts we used to build ChaNGa
    
    ```bash
    
    # ucx stuff
    export CPATH=/apps/shared/hpcx/2.4.0/ucx/include:$CPATH
    export INCLUDE=/apps/shared/hpcx/2.4.0/ucx/include:$INCLUDE
    export LD_LIBRARY_PATH=/apps/shared/hpcx/2.4.0/ucx/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/apps/shared/hpcx/2.4.0/ucx/lib/:$LIBRARY_PATH
    
    source /apps/shared/hpcx/2.4.0/hpcx-init-ompi.sh
    hpcx_load
    
    # cuda stuff
    export CUDA_HOME=/usr/local/cuda/
    export CUDATOOLKIT_HOME=$CUDA_HOME
    export CUDA_DIR=$CUDA_HOME
    export PATH=$PATH:/usr/local/cuda/bin/
    export PATH=$PATH:/home/mbeukman/utils/git/bin/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export INCLUDE=$INCLUDE:/usr/local/cuda/include/
    
    export CC=icc
    export CXX=icpc
    export MPICC=mpiicc
    export MPICXX=mpiicpc
    export MPIRUN=mpirun
    echo "========================="
    echo "        Copying          "
    echo "========================="
    cp -r  ~/downloads/changa/ .
    echo "========================="
    echo "       Making Changa     "
    echo "========================="
    cd ../changa
    # configure with cuda
    ./configure --enable-gpu-local-tree-walk \
    --with-cuda=$CUDATOOLKIT_HOME --with-cuda-level=70 2>&1 | tee config.log
    
    make -j 28 2>&1 | tee make.log
    ```
    

- A simple build script template could be the following (which logs the output to files using `tee`)
    
    ```bash
    ./configure --prefix=/home/name/software/abc 2>&1 | tee myconfigure.log
    make -j 2>&1 | tee make.log
    make install 2>&1 | tee makeinstall.log
    ```
    

# Paths and Environment Variables

Paths are important. What are they?

These are environment variables that affect where linux searches for programs when you type in only their name, or where programs look for libraries.

The main ones that are important are:

- `PATH`: Influences mainly which compilers are used
- `LD_LIBRARY_PATH`: This affects where library files are found and looked for.
- `CC, CXX, FC, F90, F77`: All of these (usually) affect which compiler is used. They correspond to the **C** **C**ompiler, the C++ Compiler, and a few different versions of the **F**ortran **C**ompiler.

You can set the path variables as follows:

`export PATH=/home/myname/path/to/software/bin:$PATH`

If you run `echo $PATH`, you will see a list of `:` separated paths, and they are searched from first to last. So if you want to be sure your program is found first instead of an identically named one elsewhere, be sure to add your path at the beginning, as above.

Usually, the value in `PATH` should be the parent directory of the program you want to find. For example, `which gcc` returns `/usr/bin/gcc`, so the thing in the `PATH` should be `/usr/bin/`

## Libraries

Libraries are usually compiled pieces of code that provide some functionality. These are mostly classified into static libraries or dynamic libraries. They perform the same function, but the method is slightly different.

- Static libraries (usually `libsomething.a`) are basically copied and pasted into the output binary during compilation, so you don't need the `.a` file when running it.
- Dynamic Libraries (usually `libsomething.so`), in contrast, are only linked, and they are then also needed when running.

**For the above reasons, it is often very useful to have a build and run script, that set the same environment variables, to be sure that you use the same libraries during building and running. Otherwise bad things can happen.**

# Optimising

Optimising applications is a central part of HPC, and it has many facets. Many techniques are quite software specific, but there are a few general categories of stuff to investigate.

### Compiler Flags

- Compiler flags can often massively increase the speed and performance of an application. There are many options (and they differ slightly from compiler to compiler). A few common ones are:
    - `-O2` or `-O3` → Turns on the most compiler optimisations. `-O3` might be somewhat unstable and may produce inaccurate results, so just validate that the results are correct before committing to this.
    - `-march=native` → Uses the best instructions that are specific to the hardware that you are compiling on. Very useful for high powered HPC hardware that have more advanced instruction sets.
    - **Be sure to compile on the same machine (or processor type) you will be running on**

### Libraries

- Most programs use libraries in some form or another. These libraries must often be provided by the user, and if you aren't careful, the default system ones (which are probably out of date and slow) will be used.
- Usually you need to compile the libraries themselves from source (and what if they depend on even more libraries? → Fun times 🙃) and then link to them in your build script.
    - This linking could be using the `PATH` or `LD_LIBRARY_PATH` variables (e.g. `export PATH=/home/myname/fftw/bin:$PATH` and `export LD_LIBRARY_PATH=/home/myname/fftw/lib:$LD_LIBRARY_PATH`)
    - Or sometimes configure scripts provide flags for this, e.g. `./configure --with-fftw=/home/myname/fftw/`
    - Or sometimes you need to do something odd and manually edit the makefile and add in a flag like `-L/home/myname/fftw/lib -lfftw` (the `-L` says where to look for libraries (if LD_LIBRARY_PATH doesn't work) and `-l` says which library to link (it looks for something of the form `libfftw.so` or `libfftw.a`

### Compilers

- What compiler you use can often times impact how fast the code runs. `GCC` is often the standard, but the Intel Compilers (you can download them using a free student license) are oftentimes much faster.
- There is also other things to consider, like which version (e.g. `gcc-5` vs `gcc-10` or `icc-2018` vs `icc-2021`).  It is usually a good idea to experiment with a bunch of different compilers, and see which one works the best for your specific software.

# Summary and Tips

So in summary, **use build and run scripts**, where the structure is something like:

- env.sh
    
    ```bash
    export PATH=/path/to/compilers/:$PATH
    export LD_LIBRARY_PATH=/path/to/lib1:/path/to/lib2:$PATH
    export CC=gcc
    export CXX=g++
    
    export FC=gfortran
    export F77=$FC
    export F90=$FC
    ```
    
- build.sh
    
    ```bash
    source env.sh
    git clone ...
    cd code
    ./configure --prefix=/home/name/software/abc 2>&1 | tee myconfig.log
    make 2>&1 | tee mymake.log
    make install 2>&1 | tee mymakeinstall.log
    
    ```
    
- run.sh
    
    ```bash
    source env.sh
    cd /home/name/software/abc/
    ./bin/software 2>&1 | tee run_today.log
    ```
    

Also, no two softwares have exactly the same structure or build process, so definitely read the `INSTALL` file, or `README` or the installation instructions on their website / documentation before doing anything. `./configure --help` is also very useful. 

For optimisation, compile something first in its most basic form, and then use that as a baseline to compare performance against when making any changes.

# Problems and Issues

You **will** have problems compiling something, and it's useful to know how to go about solving them. 

**First of all, document what you are doing!** This saves you and anyone that works with you loads of time and sanity. The build scripts above are quite useful, as they act as a record of what you have done.

**Read the logs.** When something fails, read the error messages. They will usually be found in either the `myconfigure.log` or `mymake.log` files (or wherever you redirected the error output to).  You can try searching for the word 'error' (e.g. `grep -i error myconfigure.log`) to see where something went wrong. 

Many times this error is somewhat self explanatory, like 

- "file `mylib.so` not found" (then add its parent directory to the `-L` flag or add it to `LD_LIBRARY_PATH`)

Other times it is not so much. You can google the error, and forums / mailing lists often have useful answers. 

Other common issues are not using the correct compiler / version (e.g. something is only supported in gcc-5, but you are using gcc-7 or vice versa), or language standard (e.g. the code was make for `-std=c++14` and you try and compile it with `-std=c++11`)

**Always cleanup when building again!** Whenever something fails, it can somewhat corrupt the state of the code, and it could affect your subsequent compiles. Usually `make clean`, followed by running `./configure` again solves this, but the most reliable way is to download a copy of the code again (or copy it from some archive on disk)

**If you are stuck, search around a bit first for information / ideas**: Chances are that whoever you ask for help won't exactly know how to solve your problem. 

Their first step will be to read the error message, google the error, and do research, which you can do too! You learn the most from searching for information on how to solve a problem, so only ask after you have made an effort, otherwise you won't learn anything!

# Worked Example

Let's compile nload

- [https://github.com/rolandriegel/nload](https://github.com/rolandriegel/nload/releases)

How do we go about this?

1. Get the code

```bash
wget https://github.com/rolandriegel/nload/archive/refs/tags/v0.7.4.tar.gz
```

2. Extract it

```bash
tar -xzf v0.7.4.tar.gz
```

3. cd to the code

```bash
cd nload-0.7.4
```

4. Run `./run_autools`

```bash
./run_autotools
```

5. Run configure (change the prefix)

```bash
export PREFIX=/home/myname/compile_from_source/nload
./configure --prefix=$PREFIX
```

6. make

```bash
make
```

7. make install

```bash
make install
```

8. Add it to your path

```bash
export PATH=$PREFIX/bin:$PATH
```

9. And try it out:

```bash
nload
```

# Exercises

Practice definitely helps here, and the second round of the HPC competition is basically all about this.

Some things to try:

- Compile something (anything) from source. Some simple program could be
    - Tree ([https://github.com/nodakai/tree-command](https://github.com/nodakai/tree-command))
    - htop ([https://github.com/htop-dev/htop](https://github.com/htop-dev/htop)) → Will teach you about libraries
- If you have (lots of) time, then feel free to try compile
    - Firefox ([https://davidwalsh.name/how-to-build-firefox](https://davidwalsh.name/how-to-build-firefox))
    - GCC ([https://gcc.gnu.org/wiki/InstallingGCC](https://gcc.gnu.org/wiki/InstallingGCC))
    

Then, some actual programs that are used in HPC

### FFMPEG → Image/Video processing library

[https://ffmpeg.org/](https://ffmpeg.org/)

### FFTW → Fourier Transforms

[http://www.fftw.org/](http://www.fftw.org/)

This software is used in many programs to compute the Fourier Transform, and it's useful to have some experience in compiling it.

## Proper HPC Applications

Here are some actual HPC applications that you can try and compile. I'd recommend the [HPC advisory council site](https://hpcadvisorycouncil.atlassian.net/wiki/spaces/HPCWORKS/overview?mode=global) for these ones specifically, as they also show you how to run a benchmark and measure performance.

- Gromacs
- ChaNGa
- LAMMPS

And there are many others.

# Sources

Victor Eijkhout: [Chapter on Compilers](https://pages.tacc.utexas.edu/~eijkhout/istc/html/compile.html)

# Resources

The HPC Advisory Council usually have nice tutorials on how to get started with a program

- [https://hpcadvisorycouncil.atlassian.net/wiki/spaces/HPCWORKS/pages/143982604/Applications](https://hpcadvisorycouncil.atlassian.net/wiki/spaces/HPCWORKS/pages/143982604/Applications)
- Or Google `hpc advisory council <application>`

Otherwise, the documentation on the software's webpage / Github repo is also often times useful, as it (hopefully) details some factors / flags that affect performance, and give some general tips.