#if !defined(LAYER_HPP)
#define LAYER_HPP

#include <vector>
#include <eigen3/Eigen/Dense>

using namespace std;

class Layer{
public:
    Layer(const int& num_neurons, const int& num_inputs);
    Eigen::MatrixXd biases;
    Eigen::VectorXd weights;
private:
    int num_neurons;
    // vector<float> biases;
    // vector<float> weights;
};


#endif // LAYER_HPP
