P1=generator

OMPFLAG=-fopenmp
CC=g++
CFLAGS = -g -Wall

all: generator $(P1) $(P2) $(P3)

$(P1): $(P1).cpp
	$(CC) $(CFLAGS) $(OMPFLAG) $(P1).cpp -o $(P1)

clean:
	rm -vf $(P1)
