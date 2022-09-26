CXX=mpicc

all: 1.run 2.run 3.run

%.run: %.c
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm *.run