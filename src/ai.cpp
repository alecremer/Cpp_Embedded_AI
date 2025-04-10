#include <iostream>
#include "ai.hpp"
#include "layer.hpp"
#include <eigen3/Eigen/Dense>
using namespace std;

AI::AI(const vector<int>& topology){

    int last_layer_size = topology.at(0);
    for (vector<int>::const_iterator it = topology.begin(); it != topology.end(); it++){
        Layer layer(*it, last_layer_size);
        layers.push_back(layer);
        last_layer_size = *it;
    }
}

AI::~AI(){
    cout << "AI destroyed" << endl;
}

float AI::sigmoid(const float& x){
    return (1.0f / (1.0f + exp(-x)));
}

Eigen::VectorXd AI::feed_forward(const vector<float>& input){

    // input vector to eigen
    Eigen::VectorXd input_vector(input.size());
    for (int i = 0; i < input.size(); i++){
        input_vector(i) = input[i];
    }

    // input_next_layer = sigmoid(input_vector * weights + biases)
    for(auto layer : layers){ // for all layers
        Eigen::VectorXd input_next_layer;
        for(int i = 0; i < layer.biases.rows(); i++){ // for one layer
            input_next_layer(i) = sigmoid(layer.weights.row(i).dot(input_vector) + layer.biases(i)); // for one neuron
        }
        input_vector = input_next_layer;
    }

    return input_vector;
}