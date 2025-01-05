#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// Data struct to hold inputs and targets
struct Data {
    std::vector<std::vector<int>> inputs;
    std::vector<int> targets;
};

// Neural Network class
class NeuralNetwork {
public:
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> biases;

    NeuralNetwork(const std::vector<int>& layerSizes);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& inputs, const std::vector<float>& targets, float learningRate);

private:
    std::vector<std::vector<float>> randomMatrix(int rows, int cols);
    std::vector<float> randomVector(int size);
    std::vector<float> activate(const std::vector<float>& vec);
    std::vector<float> activationGradient(const std::vector<float>& vec);
    std::vector<float> addVectors(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> dotProduct(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec);
    std::vector<std::vector<float>> outerProduct(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> hadamardProduct(const std::vector<float>& a, const std::vector<float>& b);
};

// Constructor to initialize weights and biases
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        weights.push_back(randomMatrix(layerSizes[i + 1], layerSizes[i]));
        biases.push_back(randomVector(layerSizes[i + 1]));
    }
}

// Random matrix generator
std::vector<std::vector<float>> NeuralNetwork::randomMatrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    for (auto& row : matrix) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }
    return matrix;
}

// Random vector generator
std::vector<float> NeuralNetwork::randomVector(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<float> vec(size);
    for (auto& val : vec) {
        val = dis(gen);
    }
    return vec;
}

// Forward pass
std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    std::vector<float> activation = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        activation = activate(addVectors(dotProduct(weights[i], activation), biases[i]));
    }
    return activation;
}

// Backpropagation
void NeuralNetwork::backward(const std::vector<float>& inputs, const std::vector<float>& targets, float learningRate) {
    std::vector<std::vector<float>> activations = {inputs};
    std::vector<std::vector<float>> zValues;

    std::vector<float> currentActivation = inputs;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<float> z = addVectors(dotProduct(weights[i], currentActivation), biases[i]);
        zValues.push_back(z);
        currentActivation = activate(z);
        activations.push_back(currentActivation);
    }

    std::vector<float> delta = hadamardProduct(activations.back(), targets);

    for (int i = weights.size() - 1; i >= 0; --i) {
        std::vector<std::vector<float>> weightGradient = outerProduct(delta, activations[i]);
        std::vector<float> biasGradient = delta;

        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] -= learningRate * weightGradient[j][k];
            }
            biases[i][j] -= learningRate * biasGradient[j];
        }

        if (i > 0) {
            delta = hadamardProduct(dotProduct(weights[i], delta), activationGradient(zValues[i - 1]));
        }
    }
}

// Activation functions
std::vector<float> NeuralNetwork::activate(const std::vector<float>& vec) {
    std::vector<float> activated(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        activated[i] = std::max(0.0f, vec[i]);
    }
    return activated;
}

std::vector<float> NeuralNetwork::activationGradient(const std::vector<float>& vec) {
    std::vector<float> gradient(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        gradient[i] = vec[i] > 0.0f ? 1.0f : 0.0f;
    }
    return gradient;
}

// Utility functions
std::vector<float> NeuralNetwork::addVectors(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<float> NeuralNetwork::dotProduct(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    std::vector<float> result(matrix.size());
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<std::vector<float>> NeuralNetwork::outerProduct(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(b.size()));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

std::vector<float> NeuralNetwork::hadamardProduct(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

// Data splitting
Data splitDataset(const Data& dataset, float trainRatio) {
    size_t trainSize = dataset.inputs.size() * trainRatio;
    Data trainData, valData;

    trainData.inputs = std::vector<std::vector<int>>(dataset.inputs.begin(), dataset.inputs.begin() + trainSize);
    trainData.targets = std::vector<int>(dataset.targets.begin(), dataset.targets.begin() + trainSize);

    valData.inputs = std::vector<std::vector<int>>(dataset.inputs.begin() + trainSize, dataset.inputs.end());
    valData.targets = std::vector<int>(dataset.targets.begin() + trainSize, dataset.targets.end());

    return {trainData, valData};
}

// Normalize input
std::vector<float> normalize(const std::vector<int>& input) {
    std::vector<float> normalized(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        normalized[i] = input[i] / 100.0f;
    }
    return normalized;
}

// Training
void train(NeuralNetwork& model, const Data& trainData, int epochs, float learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;

        for (size_t i = 0; i < trainData.inputs.size(); ++i) {
            auto input = normalize(trainData.inputs[i]);
            auto target = std::vector<float>(trainData.targets.size(), 0.0f);
            target[trainData.targets[i]] = 1.0f; // One-hot encoding

            auto prediction = model.forward(input);
            totalLoss += computeLoss(prediction, trainData.targets[i]);

            model.backward(input, target, learningRate);
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / trainData.inputs.size() << "\n";
    }
}

// Evaluation
void evaluate(const NeuralNetwork& model, const Data& valData) {
    float correct = 0;

    for (size_t i = 0; i < valData.inputs.size(); ++i) {
        auto input = normalize(valData.inputs[i]);
        auto prediction = model.forward(input);

        int predictedClass = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        if (predictedClass == valData.targets[i]) {
            correct++;
        }
    }

    std::cout << "Accuracy: " << (correct / valData.inputs.size()) * 100 << "%\n";
}

// Main function (Example usage)
int main() {
    // Example dataset (inputs and targets)
    Data dataset = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {0, 1, 2}};

    auto [trainData, valData] = splitDataset(dataset, 0.8);

    // Create the model
    NeuralNetwork model({3, 5, 3});

    // Train the model
    train(model, trainData, 10, 0.01);

    // Evaluate the model
    evaluate(model, valData);

    return 0;
}
