CXX=g++
CUD=nvcc
LDFLAGS=-lcudart
CXXFLAGS=-O3 -Wall -Winline -Wextra -Wno-strict-aliasing -std=c++11
CUDFLAGS=--fmad=false -O3 -arch=compute_75 -code=sm_75 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing -std=c++11

main: main.o simParams.o Grid.o
	$(CUD) $^ $(LDFLAGS) $(CUDFLAGS) -o $@

main.o: main.cu mp1-util.h simParams.h Grid.h gpuStencil.cu BC.h
	$(CUD) -c $< $(CUDFLAGS)

simParams.o: simParams.cpp simParams.h
	$(CXX) -c $< $(CXXFLAGS)

Grid.o: Grid.cu Grid.h
	$(CUD) -c $< $(CUDFLAGS)

clean:
	rm -f *.o *~ *~ *Errors.txt main
