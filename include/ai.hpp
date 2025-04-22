#ifndef AI_HPP

#define AI_HPP

#include <iostream>
#include <vector>
#include <layer.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;

class AI {
public:
    AI(const vector<int>& topology);
    ~AI();

#ifdef UNIT_TEST
    public:
#else
    private:
#endif
    Eigen::VectorXf feed_forward(const vector<float>& input);
    void backpropagation(const int& epochs, const int& batch_size, const vector<float>& input, const vector<float>& target);
    float sigmoid(const float& x);
    float sigmoid_derivative(const float& x);
    float loss_MSR(const float& x, const float& y);
    float dmsr_dy(const float& x, const float& y);
    vector<Layer> layers;
};


#endif // AI_HPP
