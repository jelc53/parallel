#ifndef TESTS_Q2_H
#define TESTS_Q2_H

#include "test_macros.h"
#include "test_util.h"

constexpr const uint kSizeMaskTest = 8;
constexpr const uint kStartBitTest = 0;

void Test1()
{
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/blockhistograms");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> blockHistograms = computeBlockHistograms(input, numBlocks,
                                        numBuckets, kSizeMaskTest, kStartBitTest, blockSize);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, blockHistograms, &success);
    PRINT_SUCCESS(success);
}

void Test2() 
{
    std::vector<uint> blockHistograms = ReadVectorFromFile("test_files/blockhistograms");
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/globalhisto");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms,
                                    numBlocks, numBuckets);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, globalHisto, &success);
    PRINT_SUCCESS(success);
}

void Test3() 
{
    std::vector<uint> globalHisto = ReadVectorFromFile("test_files/globalhisto");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/globalhistoexscan");

    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, globalHistoExScan, &success);
    PRINT_SUCCESS(success);
}

void Test4() 
{
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/blockexscan");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> globalHistoExScan = ReadVectorFromFile("test_files/globalhistoexscan");
    std::vector<uint> blockHistograms = ReadVectorFromFile("test_files/blockhistograms");
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets,
                                    numBlocks, globalHistoExScan, blockHistograms);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, blockExScan, &success);
    PRINT_SUCCESS(success);
}

void Test5() 
{
    std::vector<uint> blockExScan = ReadVectorFromFile("test_files/blockexscan");
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/sorted");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> sorted(input.size());
    populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, kStartBitTest,
                                  kSizeMaskTest, blockSize, input, sorted);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, sorted, &success);
    PRINT_SUCCESS(success);
}

#endif /* TESTS_Q2_H */
