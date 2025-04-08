#if !defined(LAYER_HPP)
#define LAYER_HPP

#include <vector>

using namespace std;

class Layer{
public:
    Layer(int num_neurons);
private:
    const int num_neurons;
    const vector<int> biases;
    const vector<int> weights;
};


#endif // LAYER_HPP
