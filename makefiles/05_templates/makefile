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
