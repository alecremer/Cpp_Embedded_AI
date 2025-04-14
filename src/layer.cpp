#include "layer.hpp"

using namespace std;

Layer::Layer(const int& num_neurons, const int& num_inputs){

    this->num_neurons = num_neurons;

    this->biases.resize(num_neurons, num_inputs);
    this->weights.resize(num_neurons, 1);
   

}
