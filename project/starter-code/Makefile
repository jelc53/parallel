OBJDIR=obj
INCDIR=inc
DEPDIR=dep
CUDASAMPLES=/usr/local/cuda-11.0/samples/common/inc

# Compilers
CC=mpic++
CUD=nvcc

TARGET := main

CPPSRCS := $(wildcard *.cpp)
CPPOBJS := $(CPPSRCS:%.cpp=$(OBJDIR)/%.o)

CUDSRCS := $(wildcard *.cu)
CUDOBJS := $(CUDSRCS:%.cu=$(OBJDIR)/%.o)

UTLSRCS := $(wildcard utils/*.cpp) 
UTLOBJS := $(UTLSRCS:utils/%.cpp=$(OBJDIR)/%.o)

OBJS := $(CPPOBJS) $(CUDOBJS) $(UTLOBJS)

DEPFILES := $(OBJS:$(OBJDIR)/%.o=$(DEPDIR)/%.d)

# Flags
CFLAGS=-O3 -std=c++11 -DARMA_DONT_USE_WRAPPER -DARMA_USE_LAPACK
CUDFLAGS=-c -O3 -arch=compute_75 -code=sm_75 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing
INCFLAGS=-I$(CUDASAMPLES) -I$(INCDIR)
LDFLAGS=-lblas -llapack -larmadillo -lcublas -lcudart
DEPFLAGS=-MT $@ -MMD -MF $(addprefix $(DEPDIR)/, $(notdir $*)).d

CC_CMD=$(CC) $(CFLAGS) $(INCFLAGS)
CU_CMD=$(CUD) $(CUDFLAGS) $(INCFLAGS)

# --fmad=false

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(CPPOBJS): $(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CC_CMD) -c $< -o $@ $(DEPFLAGS)

$(UTLOBJS): $(OBJDIR)/%.o: utils/%.cpp $(DEPDIR)/%.d 
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CC_CMD) -c $< -o $@ $(DEPFLAGS)

$(CUDOBJS): $(OBJDIR)/%.o: %.cu $(DEPDIR)/%.d
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CU_CMD) -c $< -o $@ $(DEPFLAGS)

$(DEPFILES):
include $(wildcard $(DEPFILES))

clean:
	rm -rf $(OBJDIR)/*.o $(DEPDIR)/*.d main

clear:
	rm -rf fp-* 

dev_clean:
	rm -rf $(OBJDIR)/*.o $(DEPDIR)/*.d main
	rm -rf fp-*
	rm -f Outputs/CPUmats/*
	rm -rf Debug*
