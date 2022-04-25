CXX=nvcc
CXXFLAGS=-O3 -std=c++11 -arch=compute_75 -code=sm_75
SRC1=main_q1.cu
SRC2=main_q2.cu
SRC3=main_q3.cu
INC=.

SRC := $(wildcard *.cpp) $(wildcard *.cu)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC))) $(patsubst %.cu,%,$(filter %.cu,$(SRC)))

.PHONY: default run1 run2 run3 clean

default: $(EXE)

# pattern rules
% : %.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

run1: main_q1
	srun --partition=CME --gres=gpu:1 ./main_q1

run2: main_q2
	srun --partition=CME --gres=gpu:1 ./main_q2

run3: main_q3
	srun --partition=CME --gres=gpu:1 ./main_q3

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q3
	rm -rf *.dSYM
