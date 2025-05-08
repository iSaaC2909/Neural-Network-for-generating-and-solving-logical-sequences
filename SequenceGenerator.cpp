#include "Main.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>

// Function to generate a sequence of alternating numbers and letters
std::vector<std::string> generateAlternatingSequence(int start, int length) {
    std::vector<std::string> sequence;
    for (int i = 0; i < length; ++i) {
        if (i % 2 == 0) { // Add numbers
            sequence.push_back(std::to_string(start + i / 2));
        } else { // Add letters
            sequence.push_back(std::string(1, 'A' + (i / 2) % 26));
        }
    }
    return sequence;
}

std::vector<int> encodeSequence(const std::vector<std::string>& sequence) {
    std::vector<int> encoded;
    for (const auto& element : sequence) {
        if (std::isdigit(element[0])) { // If it's a number
            encoded.push_back(std::stoi(element));
        } else if (std::isalpha(element[0])) { // If it's a letter
            encoded.push_back(element[0] - 'A' + 27); // Map A-Z to 27-52
        }
    }
    return encoded;
}

struct Data {
    std::vector<std::vector<int>> inputs;
    std::vector<int> targets;
};

Data prepareData(int numSequences, int sequenceLength, int startValue) {
    Data data;

    for (int i = 0; i < numSequences; ++i) {
        // Generate a sequence
        auto sequence = generateAlternatingSequence(startValue + i, sequenceLength);

        // Encode the sequence
        auto encoded = encodeSequence(sequence);

        // Split into inputs (all but last) and target (last)
        data.inputs.push_back(std::vector<int>(encoded.begin(), encoded.end() - 1));
        data.targets.push_back(encoded.back());
    }

    return data;
}

void saveDataset(const Data& data, const std::string& filename) {
    std::ofstream file(filename);

    for (size_t i = 0; i < data.inputs.size(); ++i) {
        for (const auto& value : data.inputs[i]) {
            file << value << ",";
        }
        file << data.targets[i] << "\n";
    }

    file.close();
}

Data loadDataset(const std::string& filename) {
    Data data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<int> input;
        int target, value;
        char comma;

        while (ss >> value) {
            if (ss.peek() == ',') ss.ignore();
            input.push_back(value);
        }

        target = input.back(); // The last value is the target
        input.pop_back();      // Remove target from inputs

        data.inputs.push_back(input);
        data.targets.push_back(target);
    }

    return data;
}

int main() {
    int numSequences = 5;   // Number of sequences to generate
    int mode = 1;           // Rule mode (1: Arithmetic, 2: Fibonacci, 3: Geometric)
    int param1 = 1;         // Start value
    int param2 = 2;         // Step or ratio
    int length = 6;         // Length of each sequence

    // Prepare dataset using complex sequences
    Data dataset = prepareDataWithComplexSequences(numSequences, mode, param1, param2, length);

    // Print sequences for verification
    for (size_t i = 0; i < dataset.inputs.size(); ++i) {
        std::cout << "Sequence " << i + 1 << ": ";
        for (const auto& val : dataset.inputs[i]) {
            std::cout << val << " ";
        }
        std::cout << "| Target: " << dataset.targets[i] << "\n";
    }
    int numSequences = 10;      // Number of sequences to generate
    int sequenceLength = 6;     // Length of each sequence
    int startValue = 1;         // Starting value for numbers

    // Generate and prepare dataset
    Data dataset = prepareData(numSequences, sequenceLength, startValue);

    // Save the dataset to a file
    saveDataset(dataset, "dataset.csv");

    // Load the dataset from the file
    Data loadedDataset = loadDataset("dataset.csv");

    // Print loaded dataset for verification
    for (size_t i = 0; i < loadedDataset.inputs.size(); ++i) {
        std::cout << "Sequence " << i + 1 << ": ";
        for (const auto& val : loadedDataset.inputs[i]) {
            std::cout << val << " ";
        }
        std::cout << "| Target: " << loadedDataset.targets[i] << "\n";
    }

    return 0;
}

std::vector<std::string> generateArithmeticSequence(int start, int step, int length) {
    std::vector<std::string> sequence;
    for (int i = 0; i < length; ++i) {
        if (i % 2 == 0) { // Numbers with arithmetic progression
            sequence.push_back(std::to_string(start + (i / 2) * step));
        } else { // Letters with arithmetic progression
            sequence.push_back(std::string(1, 'A' + ((i / 2) * step) % 26));
        }
    }
    return sequence;
}

std::vector<std::string> generateFibonacciSequence(int length) {
    std::vector<std::string> sequence;
    int a = 0, b = 1;

    for (int i = 0; i < length; ++i) {
        if (i % 2 == 0) { // Add Fibonacci numbers
            sequence.push_back(std::to_string(a));
            int next = a + b;
            a = b;
            b = next;
        } else { // Add letters (position modulo 26)
            sequence.push_back(std::string(1, 'A' + (a % 26)));
        }
    }
    return sequence;
}

std::vector<std::string> generateGeometricSequence(int start, int ratio, int length) {
    std::vector<std::string> sequence;
    int current = start;

    for (int i = 0; i < length; ++i) {
        if (i % 2 == 0) { // Add numbers with geometric progression
            sequence.push_back(std::to_string(current));
            current *= ratio;
        } else { // Add letters based on current value modulo 26
            sequence.push_back(std::string(1, 'A' + (current % 26)));
        }
    }
    return sequence;
}

std::vector<std::string> generateComplexSequence(int mode, int param1, int param2, int length) {
    switch (mode) {
        case 1: // Alternating arithmetic progression
            return generateArithmeticSequence(param1, param2, length);
        case 2: // Alternating Fibonacci sequence
            return generateFibonacciSequence(length);
        case 3: // Alternating geometric progression
            return generateGeometricSequence(param1, param2, length);
        default:
            throw std::invalid_argument("Invalid mode selected for sequence generation.");
    }
}

Data prepareDataWithComplexSequences(int numSequences, int mode, int param1, int param2, int length) {
    Data data;

    for (int i = 0; i < numSequences; ++i) {
        // Generate a complex sequence
        auto sequence = generateComplexSequence(mode, param1 + i, param2, length);

        // Encode the sequence
        auto encoded = encodeSequence(sequence);

        // Split into inputs (all but last) and target (last)
        data.inputs.push_back(std::vector<int>(encoded.begin(), encoded.end() - 1));
        data.targets.push_back(encoded.back());
    }

    return data;
}
