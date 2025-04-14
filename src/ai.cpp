#include <iostream>
#include "ai.hpp"
#include "layer.hpp"
#include <vector>
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

float AI::sigmoid_derivative(const float& x){
    return sigmoid(x)*(1.0f - sigmoid(x));
}

float AI::loss_MSR(const float& x, const float& y){
    
    return (1.0f/2.0f)*(x - y)*(x - y);

}

float AI::dmsr_dy(const float& x, const float& y){

    return (x - y);

}

void AI::backpropagation(const int& epochs, const int& batch_size, const vector<float>& input, const vector<float>& target){
    

    Eigen::VectorXd output = feed_forward(input);
    Eigen::VectorXd target_vector(input.size());
    for (int i = 0; i < target.size(); i++){
        target_vector(i) = target[i];
    }

    Eigen::VectorXd dloss_dy = output.binaryExpr(target_vector, [&](auto x, auto y){ return dmsr_dy(x, y); });

    float eta = 0.001f;
    
    float last_derivates_w = dloss_dy.sum();
    float last_derivates_b = dloss_dy.sum();

    for(int i = 0; i < epochs; i++){
        for(vector<Layer>::reverse_iterator it = layers.rbegin(); it != layers.rend(); it++){
            
            Eigen::VectorXd dw = it->weights.unaryExpr([&](auto w){ return it->activation_function_dw(w); });
            Eigen::VectorXd db = it->biases.unaryExpr([&](auto b){ return it->activation_function_db(b); });

            it->weights = it->weights - eta * (last_derivates_w * dw);
            it->biases = it->biases - eta * (last_derivates_b * db);

            last_derivates_w = (last_derivates_w * dw).sum();
            last_derivates_b = (last_derivates_b * db).sum();

        }

    }

    

}