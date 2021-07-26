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
