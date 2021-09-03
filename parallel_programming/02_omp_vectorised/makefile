CXX  ?= g++
# Get the .cpp files in this directory
SOURCES  := $(wildcard  *.cpp)
# Remove the .cpp extention
FILES    := ${SOURCES:.cpp=}
# prepend bin/ to the above files
BINS     := $(patsubst %,bin/%,$(FILES))

# We want to compile all the binaries in $(BINS)
all: $(BINS)

# This tells us how to compile each file, using c++17, avx, avx2 and fma instructions.
bin/%: %.cpp utils.h
	$(CXX) -O3 -mavx2 -mavx -mfma -std=c++17  -fopenmp $< -o $@

clean:
	rm -f bin/*
