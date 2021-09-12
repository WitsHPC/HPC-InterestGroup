CXX  ?= g++
# Potentially if this fails, use  -L/usr/local/lib -I/usr/local/include and CXX=/usr/local/Cellar/llvm/12.0.1/bin/clang++
LIBS :=  -lsfml-graphics -lsfml-window -lsfml-system -stdlib=libc++ -march=native
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
	$(CXX) -O3 -std=c++17 $(LIBS) $< -o $@

clean:
	rm -f bin/*
