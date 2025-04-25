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
    // [1, 1]
    // [2, 1]
    // [1, 2]
}

AI::~AI(){
    cout << "AI destroyed" << endl;
}

float AI::sigmoid(const float& x){
    return (1.0f / (1.0f + exp(-x)));
}

Eigen::VectorXf AI::feed_forward(const vector<float>& input){

    // input vector to eigen
    Eigen::VectorXf input_vector(input.size());
    for (int i = 0; i < input.size(); i++){
        input_vector[i] = input[i];
    }

    // input_next_layer = sigmoid(input_vector * weights + biases)
    for(auto &layer : layers){ // for all layers
        layer.inputs_cache = input_vector;
        Eigen::VectorXf input_next_layer(layer.biases.rows());
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
Eigen::VectorXf AI::vector2eigen_vector(const vector<float>& v){

    Eigen::VectorXf out(v.size());
    for (int i = 0; i < v.size(); i++){
        out(i) = v[i];
    }

    return out;
}


void AI::backpropagation(const int& epochs, const int& batch_size, const vector<float>& input, const vector<float>& target){
    
    Eigen::VectorXf target_vector = vector2eigen_vector(target);

    
    // sample value
    float eta = 0.001f;
    
    
    Eigen::VectorXf middle_propagation;
    Eigen::VectorXf w_new;
    Eigen::VectorXf b_new;
    
    for(int i = 0; i < epochs; i++){
        
        Eigen::VectorXf output = feed_forward(input);
        Eigen::VectorXf err = output.binaryExpr(target_vector, [&](auto x, auto y){return loss_MSR(x, y);});
        
        // derr/dy
        Eigen::VectorXf dloss_dy = output.binaryExpr(target_vector, [&](auto x, auto y){ return dmsr_dy(x, y); });
        
        Eigen::VectorXf dy_dz = output.unaryExpr([&](auto out){return out*(1-out);});

        for(vector<Layer>::reverse_iterator it = layers.rbegin(); it != layers.rend(); it++){
            


            
            vector<Layer>::reverse_iterator next_layer_it = prev(it);
            Eigen::VectorXf dz_da(next_layer_it->weights.rows());
            
            
            // get dz/da            
            if(next_layer_it != layers.rbegin()){
                for(int i = 0; i < next_layer_it->weights.rows(); i++){ 
                    dz_da[i] = next_layer_it->weights.row(i).sum();
                }
            }
            
            Eigen::VectorXf da_dz1 = it->inputs_cache.unaryExpr([&](auto x){ return x*(1-x); });
            
            // get new w and b
            if(it != layers.rbegin()){
                
                b_new = dloss_dy * dy_dz;
                w_new = dloss_dy * dy_dz * it->inputs_cache;
                middle_propagation = dz_da * da_dz1;
            }

            else{
                w_new = dloss_dy * dy_dz * middle_propagation * dz_da * da_dz1 * it->inputs_cache;
                b_new = dloss_dy * dy_dz * middle_propagation * dz_da * da_dz1;
            }

            // get middle propagation
            middle_propagation = middle_propagation * dz_da * da_dz1;
            
            // update b and w
            it->biases = it->biases - eta * b_new;
            it->weights = it->weights - eta * w_new;


        }

    }

    

}