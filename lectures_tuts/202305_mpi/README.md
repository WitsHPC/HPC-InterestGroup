## Tutorial: Compiling "Hello, World!" in C++ with OpenMPI and MPI
> Sayfullah Jumoorty

**Step 1: Install OpenMPI and MPI**

First, make sure you have OpenMPI and MPI installed on your system. You can download them from the official OpenMPI website (https://www.open-mpi.org/) or install them using your package manager if it's available.

**Step 2: Write the "Hello, World!" Program**

Create a new file called `hello_mpi.cpp` and open it in a text editor. Add the following code to the file:

```cpp
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    MPI_Finalize();

    return 0;
}
```

This code initializes MPI, retrieves the rank and size of the current process, and then prints a "Hello" message with the process rank and size. Finally, it cleans up MPI resources and exits.

**Step 3: Compile the Program**

To compile the program, open a terminal or command prompt and navigate to the directory where you saved `hello_mpi.cpp`. Use the following command to compile the program:

```
mpic++ hello_mpi.cpp -o hello_mpi
```

This command uses `mpic++` to compile the C++ code and produces an executable named `hello_mpi`.

**Step 4: Run the Program**

After successfully compiling the program, you can run it using the `mpirun` command. In the terminal, execute the following command:

```
mpirun -np <num_processes> ./hello_mpi
```

Replace `<num_processes>` with the number of processes you want to run. This command will launch multiple instances of the program, each representing a separate process. Each process will print its "Hello" message along with its rank and size.

That's it! You have successfully compiled and run a "Hello, World!" program using C++ with OpenMPI and MPI. You can modify the code and experiment with larger parallel programs using MPI's communication and synchronization features.