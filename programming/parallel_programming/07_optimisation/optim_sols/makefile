INC="./common"
INCFLAGS=-I$(INC) -I. -I..



all: DIR bin/serial bin/serial_O3 bin/serial_O3_copy bin/serialC bin/serialOptim bin/ompOptim bin/cuda bin/cudaOptim bin/cudaOptim2 bin/cudaOptim3


bin/serial: serial.cpp
	g++ -std=c++11 serial.cpp -o bin/serial

bin/serial_O3: serial.cpp
	g++ -O3 -std=c++11 serial.cpp -o bin/serial_O3


bin/serial_O3_copy: serial_copy_data.cpp
	g++ -O3 -std=c++11 serial_copy_data.cpp -o bin/serial_O3_copy


bin/serialC: serial.c
	gcc -O3 serial.c -o bin/serialC


bin/serialOptim: serial_optim.c
	gcc -O3 serial_optim.c -o bin/serialOptim

bin/ompOptim: openmp_optim.c
	gcc -O3 -fopenmp openmp_optim.c -o bin/ompOptim


bin/cuda: cuda.cu
	nvcc $(INCFLAGS) -I -O3 cuda.cu -o bin/cuda

bin/cudaOptim: cuda_optim.cu
	nvcc $(INCFLAGS) -I -O3 cuda_optim.cu -o bin/cudaOptim

bin/cudaOptim2: cuda_optim_2.cu
	nvcc $(INCFLAGS) -I -O3 -use_fast_math cuda_optim_2.cu -o bin/cudaOptim2

bin/cudaOptim3: cuda_optim_3.cu
	nvcc $(INCFLAGS) -I -O3 -use_fast_math cuda_optim_3.cu -o bin/cudaOptim3


DIR:
	mkdir -p bin
