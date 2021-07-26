# Parts of this makefile is based on: https://stackoverflow.com/questions/3968656/how-to-compile-mpi-and-non-mpi-version-of-the-same-program-with-automake
# The vast majority was based off of this: https://stackoverflow.com/a/27794283, which in turn references http://scottmcpeak.com/autodepend/autodepend.html.
# I used that as a starting point and simply duplicated necessary parts to accommodate compiling a CUDA and MPI Binary.
# This was also used, but later discarded: https://stackoverflow.com/questions/58575795/makefile-rule-for-depends-on-header-file-if-present.

# The idea behind this makefile is that it automatically generates .d dependency files (using something like g++ -MM),
# and uses this to determine when to recompile which file => only if it or its dependencies changed. This results in faster, incremental builds.


CXX 			:= g++
PROG 			:= main

CXXFLAGS 		:= -I./src

TARGETDIR 		:= bin
SRCDIR 			:= src
BUILDDIR 		:= obj

SRCEXT 			:= cpp
OBJEXT 			:= o
DEPEXT 			:= d

# Find me all of the source files in this directory and deeper down
SOURCE_FILES    := $(shell find src  -type f -name *.cpp)
# Now perform the following rule on these: X.cpp -> X.o
OBJECT_FILES    := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCE_FILES:.$(SRCEXT)=.$(OBJEXT)))

# bin/main: obj/main.o obj/...
$(TARGETDIR)/$(PROG): $(OBJECT_FILES)
	@printf "%-10s: linking   %-30s -> %-100s\n" $(CXX) "$^"  $(TARGETDIR)/$(TARGET)
	@mkdir -p bin
	@$(CXX) $^ -o $@

# This says that each .o file in obj/ depends on a .cpp file in the same folder structure, just inside src/
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $< -c -o $@
# Pretty output
	@printf "%-10s: compiling %-30s -> %-100s\n" $(CXX) $(shell basename $<)  $@
	
# This is somewhat magic, it uses the g++ -MM flag to generate dependencies for each file and uses sed to parse them into a proper format
	@$(CXX) $(CXXFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

# This says we can include rules from these .d files to get all the prerequisites.
-include $(OBJECT_FILES:.$(OBJEXT)=.$(DEPEXT))

# Most makefiles have a clean, which just removes some build files (*.o) and the output binary (main)
clean:
	rm -rf obj bin


.PHONY: clean
