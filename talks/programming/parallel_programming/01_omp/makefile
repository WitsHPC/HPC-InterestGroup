CXX  ?= g++
# Get the .cpp files in this directory
SOURCES  := $(wildcard  *.cpp)
# Remove the .cpp extention
FILES    := ${SOURCES:.cpp=}
# prepend bin/ to the above files
BINS     := $(patsubst %,bin/%,$(FILES))

# We want to compile all the binaries in $(BINS)
all: $(BINS)

# This tells us how to compile each file.
bin/%: %.cpp
	$(CXX) -O3 -std=c++11  -fopenmp $< -o $@

clean:
	rm -f bin/*
