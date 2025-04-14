#if !defined(LAYER_HPP)
#define LAYER_HPP

#include <vector>
#include <eigen3/Eigen/Dense>
#include <functional>

using namespace std;

class Layer{
public:
    Layer(const int& num_neurons, const int& num_inputs);
    Eigen::MatrixXd biases;
    Eigen::VectorXd weights;
    function<float(float)> activation_function;
    function<float(float)> activation_function_dw;
    function<float(float)> activation_function_db;

private:
    int num_neurons;
    // vector<float> biases;
    // vector<float> weights;
};


#endif // LAYER_HPP
