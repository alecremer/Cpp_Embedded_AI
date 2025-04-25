#include <gtest/gtest.h>
#include "ai.hpp"
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace std;


TEST(NetTest, Sigmoid){
  AI ai({1, 2, 1});
  // s[1*1 +1] 
  EXPECT_NEAR(ai.sigmoid(1.0f), 0.731f, 0.001);
  EXPECT_NEAR(ai.sigmoid(32.0f), 0.999f, 0.001);
  EXPECT_NEAR(ai.sigmoid(0.001f), 0.500f, 0.001);
  EXPECT_NEAR(ai.sigmoid(-0.001f), 0.499f, 0.001);
}

TEST(CreateNet, LayersSize){
  AI ai({1, 2, 1});
  EXPECT_EQ(ai.layers.size(), 3);

}

TEST(CreateNet, BiasSize){
  AI ai({1, 2, 1});
  EXPECT_EQ(ai.layers[0].biases.size(), 1);
  EXPECT_EQ(ai.layers[1].biases.size(), 2);
  EXPECT_EQ(ai.layers[2].biases.size(), 1);

}

TEST(CreateNet, WeightsSize){
  AI ai({1, 2, 1});
  Eigen::MatrixXf cmp_size(1, 1);
  
  EXPECT_EQ(ai.layers[0].weights.size(), cmp_size.size());
  
  cmp_size.resize(2, 1);
  EXPECT_EQ(ai.layers[1].weights.size(), cmp_size.size());
  
  cmp_size.resize(1, 2);
  EXPECT_EQ(ai.layers[1].weights.size(), cmp_size.size());
  
}


TEST(FeedforwardTest, OneNet){
  AI ai({1});
  Eigen::VectorXf output = ai.feed_forward(vector<float>{1.0f});
  EXPECT_NEAR(output[0], 0.880f, 0.001f) << "out: " << output[0];

}

TEST(FeedforwardTest, OneTwoNet){
  
  AI ai({1, 2});
  Eigen::VectorXf output = ai.feed_forward(vector<float>{1.0f});

  EXPECT_NEAR(output[0], 0.867f, 0.01f) << "out: " << output[0];
  EXPECT_NEAR(output[1], 0.867f, 0.01f) << "out: " << output[0];
  // 0.86770298908501
}

TEST(FeedforwardTest, OneTwoOneNet){
  
  AI ai({1, 2, 1});
  Eigen::VectorXf output = ai.feed_forward(vector<float>{1.0f});

  EXPECT_NEAR(output[0], 0.93908348f, 0.00001f) << "out: " << output[0];
}

TEST(FeedforwardTest, OneNetOutSize){
  AI ai({1});
  Eigen::VectorXf cmp_size(1);

  Eigen::VectorXf output = ai.feed_forward(vector<float>{1.0f});
  EXPECT_EQ(output.size(), cmp_size.size());

}

TEST(FeedforwardTest, OneTwoNetOutSize){
  AI ai({1, 2});
  Eigen::VectorXf cmp_size(2);

  Eigen::VectorXf output = ai.feed_forward(vector<float>{1.0f});
  EXPECT_EQ(output.size(), cmp_size.size());

}

TEST(FeedforwardTest, OneTwoOneNetOutSize){
  AI ai({1, 2, 1});
  Eigen::VectorXf cmp_size(1);

  Eigen::VectorXf output = ai.feed_forward(vector<float>{1.0f});
  EXPECT_EQ(output.size(), cmp_size.size());

}

TEST(BackpropagationTest, LossMSR){
  AI ai({1, 2, 1});
  Eigen::VectorXf output = ai.feed_forward(vector<float>{1.0f});

  EXPECT_NEAR(ai.loss_MSR(output[0], 2.0f), 0.5625203f, 0.001f);

}

TEST(BackpropagationTest, UpdateBiasLast){
  AI ai({1, 2, 1});
  ai.backpropagation(1, 1, vector<float>{1.0f}, vector<float>{2.0f});

  EXPECT_NEAR(ai.layers[2].biases[0], -0.353623138f, 0.001f);

}

