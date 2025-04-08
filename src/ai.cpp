#include <iostream>
#include "ai.hpp"
#include "layer.hpp"
using namespace std;

AI::AI(const vector<int>& topology){

    
    for (vector<int>::const_iterator it = topology.begin(); it != topology.end(); it++){
        Layer layer(*it);
        layers.push_back(layer);
    }
}
AI::~AI(){
    cout << "AI destroyed" << endl;
}