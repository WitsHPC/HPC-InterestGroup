INC="./common"
INCFLAGS=-I$(INC) -I. -I..
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

julia: julia.cu
	$(NVCC) $(INCFLAGS) $(GLUTFLAGS) -Xcompiler -fopenmp -lgomp julia.cu -o julia

run: julia
	./julia
	@convert cuda.ppm cuda.png
	@rm cuda.ppm