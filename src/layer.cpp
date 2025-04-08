#include "layer.hpp"

using namespace std;

Layer::Layer(int num_neurons){

    this->num_neurons = num_neurons;

    biases.resize(num_neurons, 0);
    weights.resize(num_neurons, 0);
   

}
