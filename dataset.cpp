#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>

// Data structure to hold the inputs and targets
struct Data {
    std::vector<std::vector<int>> inputs; // List of input sequences
    std::vector<int> targets;             // List of target values
};

// Function to save the dataset to a file
void saveDataset(const Data& data, const std::string& filename);

// Function to load the dataset from a file
Data loadDataset(const std::string& filename);

#endif // DATASET_HPP
