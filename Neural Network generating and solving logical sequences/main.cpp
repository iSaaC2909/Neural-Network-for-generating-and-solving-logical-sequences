#include <iostream>
#include "Main.hpp"

int main() {
    // Settings for sequence generation
    int numSequences = 5;   // Number of sequences to generate
    int sequenceLength = 6;  // Length of each sequence
    int startValue = 1;      // Starting value for numbers

    // Prepare the dataset using alternating sequences of numbers and letters
    Data dataset = prepareData(numSequences, sequenceLength, startValue);

    // Print sequences for verification
    std::cout << "Generated Sequences and Targets:" << std::endl;
    for (size_t i = 0; i < dataset.inputs.size(); ++i) {
        std::cout << "Sequence " << i + 1 << ": ";
        for (const auto& val : dataset.inputs[i]) {
            std::cout << val << " ";
        }
        std::cout << "| Target: " << dataset.targets[i] << std::endl;
    }

    // Save the dataset to a CSV file
    saveDataset(dataset, "dataset.csv");
    std::cout << "\nDataset saved to 'dataset.csv'." << std::endl;

    // Load the dataset from the file
    Data loadedDataset = loadDataset("dataset.csv");

    // Print loaded dataset for verification
    std::cout << "\nLoaded Dataset:" << std::endl;
    for (size_t i = 0; i < loadedDataset.inputs.size(); ++i) {
        std::cout << "Sequence " << i + 1 << ": ";
        for (const auto& val : loadedDataset.inputs[i]) {
            std::cout << val << " ";
        }
        std::cout << "| Target: " << loadedDataset.targets[i] << std::endl;
    }

    return 0;
}
