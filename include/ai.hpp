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


private:
    Eigen::VectorXd feed_forward(const vector<float>& input);
    float sigmoid(const float& x);
    vector<Layer> layers;
};


#endif // AI_HPP
