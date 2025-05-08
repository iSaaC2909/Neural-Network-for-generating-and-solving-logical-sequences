#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>

// Structure to hold dataset
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
    void saveModel(const std::string& filePath);
    void loadModel(const std::string& filePath);

private:
    std::vector<std::vector<float>> randomMatrix(int rows, int cols);
    std::vector<float> randomVector(int size);
    std::vector<float> activate(const std::vector<float>& vec);
    std::vector<float> addVectors(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> dotProduct(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec);
    std::vector<std::vector<float>> outerProduct(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> lossGradient(const std::vector<float>& output, const std::vector<float>& target);
    std::vector<float> activationGradient(const std::vector<float>& z);
    std::vector<float> hadamardProduct(const std::vector<float>& a, const std::vector<float>& b);
};

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        weights.push_back(randomMatrix(layerSizes[i + 1], layerSizes[i]));
        biases.push_back(randomVector(layerSizes[i + 1]));
    }
}

std::vector<std::vector<float>> NeuralNetwork::randomMatrix(int rows, int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& row : matrix) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }
    return matrix;
}

std::vector<float> NeuralNetwork::randomVector(int size) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& val : vec) {
        val = dis(gen);
    }
    return vec;
}

std::vector<float> NeuralNetwork::addVectors(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<float> NeuralNetwork::dotProduct(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec) {
    std::vector<float> result(mat.size());
    for (size_t i = 0; i < mat.size(); ++i) {
        result[i] = std::inner_product(mat[i].begin(), mat[i].end(), vec.begin(), 0.0f);
    }
    return result;
}

std::vector<float> NeuralNetwork::activate(const std::vector<float>& vec) {
    std::vector<float> activated(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        activated[i] = std::max(0.0f, vec[i]); // ReLU activation
    }
    return activated;
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    std::vector<float> activation = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        activation = activate(addVectors(dotProduct(weights[i], activation), biases[i]));
    }
    return activation;
}

void NeuralNetwork::backward(const std::vector<float>& inputs, const std::vector<float>& targets, float learningRate) {
    // Store activations and weighted sums (z-values) during forward pass
    std::vector<std::vector<float>> activations = {inputs};
    std::vector<std::vector<float>> zValues;

    // Forward pass to cache activations and z-values
    std::vector<float> currentActivation = inputs;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<float> z = addVectors(dotProduct(weights[i], currentActivation), biases[i]);
        zValues.push_back(z);
        currentActivation = activate(z);
        activations.push_back(currentActivation);
    }

    // Compute the loss gradient with respect to the output
    std::vector<float> delta = lossGradient(activations.back(), targets);

    // Backpropagate through layers
    for (int i = weights.size() - 1; i >= 0; --i) {
        // Compute gradients for weights and biases
        std::vector<std::vector<float>> weightGradient = outerProduct(delta, activations[i]);
        std::vector<float> biasGradient = delta;

        // Update weights and biases using gradient descent
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] -= learningRate * weightGradient[j][k];
            }
            biases[i][j] -= learningRate * biasGradient[j];
        }

        // Compute delta for the previous layer
        if (i > 0) {
            delta = hadamardProduct(dotProduct(transpose(weights[i]), delta), activationGradient(zValues[i - 1]));
        }
    }
}

void NeuralNetwork::saveModel(const std::string& filePath) {
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for saving model." << std::endl;
        return;
    }

    for (const auto& layerWeights : weights) {
        for (const auto& neuronWeights : layerWeights) {
            for (float weight : neuronWeights) {
                file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            }
        }
    }

    for (const auto& layerBiases : biases) {
        for (float bias : layerBiases) {
            file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
        }
    }

    file.close();
}

void NeuralNetwork::loadModel(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for loading model." << std::endl;
        return;
    }

    for (auto& layerWeights : weights) {
        for (auto& neuronWeights : layerWeights) {
            for (float& weight : neuronWeights) {
                file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            }
        }
    }

    for (auto& layerBiases : biases) {
        for (float& bias : layerBiases) {
            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        }
    }

    file.close();
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
    std::vector<std::vector<float>> result(mat[0].size(), std::vector<float>(mat.size()));
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

std::vector<float> NeuralNetwork::lossGradient(const std::vector<float>& output, const std::vector<float>& target) {
    std::vector<float> gradient(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[i] = 2.0f * (output[i] - target[i]);
    }
    return gradient;
}

std::vector<float> NeuralNetwork::activationGradient(const std::vector<float>& z) {
    std::vector<float> gradient(z.size());
    for (size_t i = 0; i < z.size(); ++i) {
        gradient[i] = z[i] > 0.0f ? 1.0f : 0.0f;
    }
    return gradient;
}

std::vector<float> NeuralNetwork::hadamardProduct(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
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

Data splitDataset(const Data& dataset, float trainRatio) {
    size_t trainSize = dataset.inputs.size() * trainRatio;
    Data trainData, valData;

    trainData.inputs = std::vector<std::vector<int>>(dataset.inputs.begin(), dataset.inputs.begin() + trainSize);
    trainData.targets = std::vector<int>(dataset.targets.begin(), dataset.targets.begin() + trainSize);

    valData.inputs = std::vector<std::vector<int>>(dataset.inputs.begin() + trainSize, dataset.inputs.end());
    valData.targets = std::vector<int>(dataset.targets.begin() + trainSize, dataset.targets.end());

    return {trainData, valData};
}

std::vector<float> normalize(const std::vector<int>& input) {
    std::vector<float> normalized(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        normalized[i] = input[i] / 100.0f; // Example normalization
    }
    return normalized;
}

float computeLoss(const std::vector<float>& prediction, const std::vector<float>& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < prediction.size(); ++i) {
        loss += std::pow(prediction[i] - target[i], 2);
    }
    return loss;
}

void train(NeuralNetwork& model, const Data& trainData, int epochs, float learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;

        for (size_t i = 0; i < trainData.inputs.size(); ++i) {
            auto input = normalize(trainData.inputs[i]);
            std::vector<float> target(trainData.targets.begin() + i, trainData.targets.begin() + i + 1);

            auto prediction = model.forward(input);
            float loss = computeLoss(prediction, target);
            totalLoss += loss;

            model.backward(input, target, learningRate);
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / trainData.inputs.size() << "\n";
    }
}

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

int main() {
    // Example usage of the neural network
    Data dataset = {
        {{1, 0, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}}, // Inputs
        {0, 1, 1, 0}                                  // Targets
    };

    float trainRatio = 0.8f;
    auto [trainData, valData] = splitDataset(dataset, trainRatio);

    std::vector<int> layerSizes = {3, 4, 2, 1};
    NeuralNetwork model(layerSizes);

    int epochs = 10;
    float learningRate = 0.01f;
    train(model, trainData, epochs, learningRate);

    evaluate
