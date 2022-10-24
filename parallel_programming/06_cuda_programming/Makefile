INC="./common"
INCFLAGS=-I$(INC) -I. -I..
CC=gcc
NVCC=nvcc

main: main.cu
	$(NVCC) $(INCFLAGS) main.cu -o main

run: main
	./main