#if !defined(LAYER_HPP)
#define LAYER_HPP

#include <vector>

using namespace std;

class Layer{
public:
    Layer(int num_neurons);
private:
    int num_neurons;
    vector<int> biases;
    vector<int> weights;
};


#endif // LAYER_HPP
