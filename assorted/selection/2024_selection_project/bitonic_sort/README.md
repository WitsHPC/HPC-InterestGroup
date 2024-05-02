Credits to Sayfullah Jumoorty and Muhammed Muaaz Dawood.

# Compiling bitonic_sort

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

# Running bitonic_sort

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
