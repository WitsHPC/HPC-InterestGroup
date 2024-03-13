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
