INC="./common"
INCFLAGS=-I$(INC) -I. -I..
CC=gcc
NVCC=nvcc

all: vector_addition matrix_mult


vector_addition: vector_addition.cu DIR
	$(NVCC) $(INCFLAGS) vector_addition.cu -o bin/vector_addition

matrix_mult: matrix_mult.cu DIR
	$(NVCC) $(INCFLAGS) matrix_mult.cu -o bin/matrix_mult


DIR: bin
bin:
	mkdir bin
run: main
	./main