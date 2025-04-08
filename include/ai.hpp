#ifndef AI_HPP

#define AI_HPP

#include <iostream>
#include <vector>
#include <layer.hpp>

using namespace std;

class AI {
public:
    AI(const vector<int>& topology);
    ~AI();

private:
    vector<Layer> layers;
};


#endif // AI_HPP
