#include "layer.hpp"

using namespace std;

Layer::Layer(const int& num_neurons, const int& num_inputs){

    this->num_neurons = num_neurons;

    this->weights.resize(num_neurons, num_inputs);
    this->biases.resize(num_neurons, 1);

    // initial value
    this->biases.setConstant(1.0f);
    this->weights.setConstant(1.0f);
   

}
