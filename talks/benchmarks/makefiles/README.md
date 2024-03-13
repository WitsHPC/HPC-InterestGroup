# Makefiles
- [Makefiles](#makefiles)
- [Intro](#intro)
  - [Speed](#speed)
  - [Incremental compilation](#incremental-compilation)
- [Make](#make)
- [Examples](#examples)
  - [Simple Example](#simple-example)
  - [Incremental](#incremental)
  - [Variables](#variables)
  - [Automatic Variables](#automatic-variables)
  - [Template rules](#template-rules)
  - [Wildcards](#wildcards)
  - [Structure and dependency](#structure-and-dependency)
- [Other methods](#other-methods)
- [Conclusion](#conclusion)
# Intro
Building code is a super common step. From compiling your own Fortran or C/C++ code, to compiling typescript to javascript, to using other peoples software, and a whole host of other things.

We'll mostly consider compiling C code, and some considerations that are important to make with respect to that.

There are many different ways to go about this process, and we will start off simple, and then see how the complexities naturally evolve.

To compile a single file, that's relatively easy:

```latex
gcc myfile.c -o myprog
```

But, what if we use flags?

```latex
gcc myfile.c -O3 -std=c99 -DFLAG -fno-omit-frame-pointer -march=native -I/my/include -I/my/other/include -o myprog
```

Kind of annoying to retype it, but what's the problem? We can simply press the up arrow.

What if you have multiple files?

Easy, just compile them at once

```latex
gcc myfile.c myfile1.c myfile2.c myfile3.c -o myprog
```

What if some files need to be compiled using different flags, and it's not just one command to use the up arrow on?

Easy, shell script

```bash
# build.sh

gcc myfile1.c $(FLAGS1) -c -o file1.o
gcc myfile2.c $(FLAGS2) -c -o file2.o

# link
gcc file1.o file2.o -o myprog
```

## Speed

If you only compile something once, the above is fine. But, what if you want to change some files, add features, etc?

Some large projects can take **hours** to fully compile (have a go at compiling gcc from source if you're skeptical)

## Incremental compilation

**Compile everything to a .o file first**

Then, only recompile files that change (which should be relatively few) and just relink.

This is the main idea behind Make.
# Make

Make is a way to compile and build large codebases, but do so in an efficient way as to have faster builds.

The key feature of make is that you can use it to only recompiles files that have changed, which can save a lot of time in large projects. 

Make is used extensively in lots of different HPC applications, as well as general software. It's quite useful to know a little bit of how exactly it works.

# Examples

This will mainly go over multiple examples, adding some features in each one.

## Simple Example

- `main.cpp`

    ```cpp
    #include <iostream>
    #include "mymaths.h"

    int main(){
        int a = 5;
        int b = 7;
        int c = add(a, b);
        std::cout << a<< " + " << b << " = " << c << "\n";
    }
    ```

- `mymaths.h`

    ```cpp
    int add(int, int);
    ```

- `mymaths.cpp`

    ```cpp
    #include "mymaths.h"
    int add(int a, int b){
        return a + b;
    }
    ```

- `makefile`

    ```makefile
    main: main.cpp mymaths.cpp
    	g++ main.cpp mymaths.cpp -o main
    ```

How does this work?

there are multiple rules, where each has the form of:

```makefile
target : prerequisite(s)
<TAB>rule(s)
```

e.g. in the above one:

- `target` = `main`
- `prerequisites` = `main.cpp mymaths.cpp`
- `rule` = `g++ main.cpp mymaths.cpp -o main`

And this means, if the prerequisite has changed since last time the rule executed, then execute the rule again.

The really cool thing is you can use other targets as prerequisites, and make then recursively checks to see if anything needs to be executed.

## Incremental

At the moment, when we change any file, we need to rebuild everything. Why?

Because, if any of our prerequisites have changed (let's say main.cpp), then we need to run the rule, which compiles everything.

We can fix that by using object files, and compiling each file individually.

- Incremental makefile

    ```makefile
    # To make main, we need the two object file
    main: main.o mymaths.o
    # once we have them, we can simply link them
    	g++ main.o mymaths.o -o main

    # the main file needs to be recompiled either when main.cpp or mymaths.h changes
    main.o: main.cpp mymaths.h
    # using this rule
    	g++ main.cpp -c -o main.o

    # similarly for mymaths.cpp
    mymaths.o: mymaths.cpp mymaths.h
    	g++ mymaths.cpp -c -o mymaths.o

    # Most makefiles have a clean, which just removes some build files (*.o) and the output binary (main)
    clean:
    	rm *.o main
    ```

## Variables

Most actual makefiles use variables for a variety of things, like:

- the compiler names (e.g. I might use gcc, but you use icc - intel's compiler)
- program names

The syntax of variables is:

At the top, usually declare the variable like in a shell script

```makefile
CXX=g++
```

And use it like `$(CXX)`

- Using Variables

    ```makefile
    CXX=g++
    PROG=main
    # To make main, we need the two object file
    $(PROG): main.o mymaths.o
    # once we have them, we can simply link them
    	$(CXX) main.o mymaths.o -o $(PROG)

    # the main file needs to be recompiled either when main.cpp or mymaths.h changes
    main.o: main.cpp mymaths.h
    # using this rule
    	$(CXX) main.cpp -c -o main.o

    # similarly for mymaths.cpp
    mymaths.o: mymaths.cpp mymaths.h
    	$(CXX) mymaths.cpp -c -o mymaths.o

    # Most makefiles have a clean, which just removes some build files (*.o) and the output binary (main)
    clean:
    	rm *.o $(PROG)
    ```

And you can update the variables from the command line

- `make` results in

    ```makefile
    g++ main.cpp -c -o main.o
    g++ mymaths.cpp -c -o mymaths.o
    g++ main.o mymaths.o -o main
    ```

- `make CXX=g++-11` results in

    ```makefile
    g++-11 main.cpp -c -o main.o
    g++-11 mymaths.cpp -c -o mymaths.o
    g++-11 main.o mymaths.o -o main
    ```

## Automatic Variables

Make also provides many automatic variables, which can make the files incomprehensible, if you don't know what they mean. A few of these automatic variables are (from Eijkhout):

`$@` The target. Use this in the link line for the main program.
`$^` The list of prerequisites. Use this also in the link line for the program.
`$<` The first prerequisite. Use this in the compile commands for the individual object files.
`$*` In template rules, this matches the template part, the part corresponding to the %.

As an example, in the following block,

```makefile
main: main.cpp mymaths.h
	g++ main.cpp -c -o main.o
```

`$@` = `main`

`$^` = `main.cpp mymaths.h`

`$<` = `main.cpp`

One other symbol in makefiles is `@`

- the `@` symbol before a command simply means to suppress the command itself, and only show the output
    - So, `$(CXX) $^ -o $@` will print out the compile line, whereas `@$(CXX) $^ -o $@` won't

So, we could replace our makefile above with this one:

- Auto variables
    ```makefile
    CXX=g++
    PROG=main
    # To make main, we need the two object file
    $(PROG): main.o mymaths.o
    # once we have them, we can simply link them
        @echo "Linking Now. Target = $@. Prereqs = "$^
        $(CXX) $^ -o $@

    %.o: %.cpp mymaths.h
    # using this rule
        @echo "Building Now. Target = $@. Prereqs = $^"
        $(CXX) $< -c -o $@

    # Most makefiles have a clean, which just removes some build files (*.o) and the output binary (main)
    clean:
        rm *.o $(PROG)

    ```
## Template rules

You can see that the rules for `main.o` and `mymaths.o` are virtually identical, so is there a way to reduce duplication? Yes! Template rules.

Using template rules, you can replace

`mymaths.o: mymaths.cpp mymaths.h`

with 

`%.o: %.cpp mymaths.h` ,where the percentage sign will ensure that for each `.o` file that is marked as a prerequisite somewhere, the rule will automatically be created.

- So we can simplify the above makefile to:

    ```makefile
    CXX=g++
    PROG=main
    # To make main, we need the two object file
    $(PROG): main.o mymaths.o
    # once we have them, we can simply link them
    	@echo "Linking Now. Target = $@. Prereqs = "$^
    	$(CXX) $^ -o $@

    %.o: %.cpp mymaths.h
    # using this rule
    	@echo "Building Now. Target = $@. Prereqs = $^"
    	$(CXX) $< -c -o $@

    # Most makefiles have a clean, which just removes some build files (*.o) and the output binary (main)
    clean:
    	rm *.o $(PROG)
    ```

## Wildcards

There are a few cool ways that you can basically say to make to compile all files in the directory

`SOURCE_FILES=${wildcard *.cpp}` → `SOURCE_FILES` is now a space separated list of all the .cpp files in this directory.

`OBJECT_FILES=${patsubst %.cpp,%.o,${SOURCE_FILES}}` → Transforms X.cpp to X.o

Then you can use the `$(OBJECT_FILES)` as the prerequisites for the `$(PROG)` target.

- Updated

    ```makefile
    CXX=g++
    PROG=main
    # Find me all of the source files here
    SOURCE_FILES=${wildcard *.cpp}
    # Now perform the following rule on these: X.cpp -> X.o
    OBJECT_FILES=${patsubst %.cpp,%.o,${SOURCE_FILES}}

    # To make main, we need the two object file
    $(PROG): $(OBJECT_FILES)
    # once we have them, we can simply link them
    	@echo "Linking Now. Target = $@. Prereqs = "$^
    	@echo "Source = $(SOURCE_FILES). Objects = $(OBJECT_FILES)"
    	$(CXX) $^ -o $@

    %.o: %.cpp mymaths.h
    # using this rule
    	@echo "Building Now. Target = $@. Prereqs = $^"
    	$(CXX) $< -c -o $@

    # Most makefiles have a clean, which just removes some build files (*.o) and the output binary (main)
    clean:
    	rm *.o $(PROG)
    ```

## Structure and dependency

I usually organise my code into largely arbitrary directories that are nested very deep, and I am also not fond of the .o files cluttering my main directory.

Also, what if you have a large project structure with many `.h` files that aren't all included in the same spots? You'd like to automatically tell make: If this file, or any of the files it includes have changed, then please recompile.

This is a simplified version of a makefile that I've had success with in the past, and it does the following:

- Automatically manages dependencies (e.g. it automatically checks which files include which other ones, using the `g++ -MM` command
- Stores object files in the same directory structure as the code, just nested under `obj/`. So, `src/maths/mymaths.cpp` will be compiled to `obj/maths/mymaths.o`

- This looks like:

    ```makefile
    # Parts of this makefile is based on: https://stackoverflow.com/questions/3968656/how-to-compile-mpi-and-non-mpi-version-of-the-same-program-with-automake
    # The vast majority was based off of this: https://stackoverflow.com/a/27794283, which in turn references http://scottmcpeak.com/autodepend/autodepend.html.
    # I used that as a starting point and simply duplicated necessary parts to accommodate compiling a CUDA and MPI Binary.
    # This was also used, but later discarded: https://stackoverflow.com/questions/58575795/makefile-rule-for-depends-on-header-file-if-present.

    # The idea behind this makefile is that it automatically generates .d dependency files (using something like g++ -MM),
    # and uses this to determine when to recompile which file => only if it or its dependencies changed. This results in faster, incremental builds.

    CXX 			:= g++
    PROG 			:= main

    CXXFLAGS 		:= -I./src

    TARGETDIR 		:= bin
    SRCDIR 			:= src
    BUILDDIR 		:= obj

    SRCEXT 			:= cpp
    OBJEXT 			:= o
    DEPEXT 			:= d

    # Find me all of the source files in this directory and deeper down
    SOURCE_FILES    := $(shell find src  -type f -name *.cpp)
    # Now perform the following rule on these: X.cpp -> X.o
    OBJECT_FILES    := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCE_FILES:.$(SRCEXT)=.$(OBJEXT)))

    # bin/main: obj/main.o obj/...
    $(TARGETDIR)/$(PROG): $(OBJECT_FILES)
    	@printf "%-10s: linking   %-30s -> %-100s\n" $(CXX) "$^"  $(TARGETDIR)/$(TARGET)
    	@mkdir -p bin
    	@$(CXX) $^ -o $@

    # This says that each .o file in obj/ depends on a .cpp file in the same folder structure, just inside src/
    $(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
    	@mkdir -p $(dir $@)
    	@$(CXX) $(CXXFLAGS) $< -c -o $@
    # Pretty output
    	@printf "%-10s: compiling %-30s -> %-100s\n" $(CXX) $(shell basename $<)  $@
    	
    # This is somewhat magic, it uses the g++ -MM flag to generate dependencies for each file and uses sed to parse them into a proper format
    	@$(CXX) $(CXXFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
    	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
    	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
    	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
    	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

    # This says we can include rules from these .d files to get all the prerequisites.
    -include $(OBJECT_FILES:.$(OBJEXT)=.$(DEPEXT))

    # Most makefiles have a clean, which just removes some build files (*.o) and the output binary (main)
    clean:
    	rm -rf obj bin

    .PHONY: clean
    ```

# Other methods

Makefiles are great and all, but they do have limitations. There are lots of other options available:

- [CMake](https://cmake.org/) is a more modern build system that can generate makefiles.
- [Autotools](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html) is a set of tools that also generate makefiles in a portable way. Their main goal is to find the correct libraries, compilers, and set the configuration up correctly for the current system.

In general, when compiling from source, you will either have a CMake structure (assume you're in the source code directory), and you can usually compile using the following:

```makefile
mkdir build && cd build
cmake ..
make
make install
```

Or using an autotools based option:

```makefile
./configure
make
make install
```

# Conclusion

So, makefiles are quite useful, and I hope you understand a bit more about how they work now.

The above examples (with code) can be found in: [https://github.com/WitsHPC/HPC-InterestGroup/tree/main/talks/benchmarks/makefiles](https://github.com/WitsHPC/HPC-InterestGroup/tree/main/talks/benchmarks/makefiles)

For more information have a look at these sources. Many of these examples and ideas came from the first one, and that book is great to read about anything HPC related.

- [Victor Eijkhout's Introduction to HPC Textbook](https://pages.tacc.utexas.edu/~eijkhout/istc/html/gnumake.html)
- [http://www.gnu.org/software/make/manual/make.html](http://www.gnu.org/software/make/manual/make.html)
