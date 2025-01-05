#ifndef MAIN_HPP
#define MAIN_HPP

#include <vector>
#include <string>

// Sequence generation functions
std::vector<std::string> generateAlternatingSequence(int start, int length);
std::vector<std::string> generateArithmeticSequence(int start, int step, int length);
std::vector<std::string> generateFibonacciSequence(int length);
std::vector<std::string> generateGeometricSequence(int start, int ratio, int length);
std::vector<std::string> generateComplexSequence(int mode, int param1, int param2, int length);

// Struct to hold dataset
struct Data {
    std::vector<std::vector<int>> inputs;
    std::vector<int> targets;
};

// Data preparation functions
Data prepareData(int numSequences, int sequenceLength, int startValue);
Data prepareDataWithComplexSequences(int numSequences, int mode, int param1, int param2, int length);

// Data saving and loading functions
void saveDataset(const Data& data, const std::string& filename);
Data loadDataset(const std::string& filename);

#endif // MAIN_HPP
