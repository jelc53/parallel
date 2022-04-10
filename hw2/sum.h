#include <vector>
#include <omp.h>
#include <iostream>

std::vector<uint> serialSum(const std::vector<uint> &v) 
{
    std::vector<uint> sums(2);
    for(uint i=0; i<v.size(); i++) {
        if (v[i] % 2 == 0) {
            sums[0] += v[i];
        }
        else {
            sums[1] += v[i];
        }
    }
    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v) 
{
    omp_set_num_threads(4);
    std::vector<uint> sums(2);
    uint sum0 = 0; uint sum1 = 0;
    
    #pragma omp parallel for reduction(+:sum0) reduction(+:sum1)
    for(uint i=0; i<v.size(); i++) {
        if (v[i] % 2 == 0) {
            sum0 += v[i];
        }
        else {
            sum1 += v[i];
        }
    }
    sums[0] = sum0; sums[1] = sum1;
    return sums;
}