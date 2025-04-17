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
    Eigen::VectorXd inputs_cache;
    function<float(float, float)> activation_function;
    function<float(float, float)> activation_function_dw;
    function<float(float, float)> activation_function_db;

private:
    int num_neurons;
    // vector<float> biases;
    // vector<float> weights;
};


#endif // LAYER_HPP
