# Neural-Network-for-generating-and-solving-logical-sequences
# Sequence Generator & Dataset Handler

This project includes a sequence generator, dataset handler, and encoding logic for generating and processing alternating sequences of numbers and letters. The dataset can be saved to a CSV file and later loaded for further analysis or training purposes.

## Features

- **Sequence Generation**: Generates alternating sequences of numbers and letters in various patterns such as:
  - Alternating sequences (e.g., 1, A, 2, B, 3, C)
  - Arithmetic sequences (e.g., 1, 3, 5, 7, 9, A)
  - Fibonacci sequences (e.g., 0, A, 1, B, 1, C)
  - Geometric sequences (e.g., 1, A, 2, B, 4, C)
  
- **Encoding**: Encodes the sequences of numbers and letters into integers for machine learning tasks, with letters mapped to numbers between 27 and 52.

- **Dataset Handling**:
  - **Saving**: Saves the generated sequences to a CSV file.
  - **Loading**: Loads a dataset from a CSV file for further processing or training.

## Installation

To use this project, simply clone the repository and compile the source code.

### Clone the repository

```bash
git clone https://github.com/iSaaC2909/Neural-Network for generating and solving logical sequences.git
cd sequence-generator

compile: g++ main.cpp dataset.cpp sequencegenerator.cpp -o main_program


The main program demonstrates how to generate a dataset of sequences, save it to a CSV file, and later load it.

Generate sequences using the prepareData or prepareDataWithComplexSequences functions.
Save the dataset to a file using the saveDataset function.
Load the dataset back into memory using the loadDataset function.

example:

#include <iostream>
#include "dataset.hpp"
#include "Main.hpp"

int main() {
    // Generate a sample dataset
    Data dataset = prepareData(5, 6, 1);  // 5 sequences, each of length 6

    // Save the dataset
    saveDataset(dataset, "dataset.csv");

    // Load the dataset back
    Data loadedDataset = loadDataset("dataset.csv");

    // Print the loaded dataset
    for (size_t i = 0; i < loadedDataset.inputs.size(); ++i) {
        std::cout << "Sequence " << i + 1 << ": ";
        for (const auto& val : loadedDataset.inputs[i]) {
            std::cout << val << " ";
        }
        std::cout << "| Target: " << loadedDataset.targets[i] << std::endl;
    }

    return 0;
}

file struct:

main.cpp: The entry point of the program, which demonstrates how to generate sequences, save datasets, and load them.
dataset.hpp: Header file defining functions and data structures for saving and loading datasets.
dataset.cpp: Implementation of the dataset handling functions.
Main.hpp: Header file for the sequence generator and encoder.
sequencegenerator.cpp: Implementation of functions for generating and encoding sequences.

Example Output
After running the main program, the output will be something like this:

Sequence 1: 1 A 2 B 3 C | Target: 4
Sequence 2: 2 A 3 B 4 C | Target: 5
Sequence 3: 3 A 4 B 5 C | Target: 6
...

Additionally, a CSV file (dataset.csv) will be created with the following format:
1, A, 2, B, 3, C, 4
2, A, 3, B, 4, C, 5
3, A, 4, B, 5, C, 6
...
