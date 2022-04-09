CXX=g++
CXXFLAGS=-std=c++11 -g -Wall -O3 -fopenmp
SRC1=main_q1.cpp
SRC2=main_q2.cpp
HDR1=sum.h
HDR2=parallel_radix_sort.h

default: main_q1 main_q2 main_q2_part6
main_q1: $(SRC1) $(HDR1)
	$(CXX) $(CXXFLAGS) $< -o $@
main_q2: $(SRC2) $(HDR2)
	$(CXX) $(CXXFLAGS) $< -o $@
main_q2_part6: $(SRC2) $(HDR2)
	$(CXX) $(CXXFLAGS) $< -D QUESTION6 -o $@

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q2_part6
	rm -rf *.dSYM
