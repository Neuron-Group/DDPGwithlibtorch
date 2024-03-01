
#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include "Data.hpp"
#include "Pars.hpp"
#include "bits/stdc++.h"
#include "torch/torch.h"

class Actor : public torch::nn::Module {
public:
  Actor(Pars pars) {
    lstm = torch::nn::LSTM(
        torch::nn::LSTMOptions(pars.state_size, pars.lstm_hidden_size)
            .num_layers(pars.lstm_num_layers)
            .batch_first(true));
    lkrelu = torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(pars.leaky_relu_slope));
    fc1 = torch::nn::Linear(pars.lstm_hidden_size, pars.actor_hidden_units[0]);
    fc2 = torch::nn::Linear(pars.actor_hidden_units[0],
                            pars.actor_hidden_units[1]);
    fc3 = torch::nn::Linear(pars.actor_hidden_units[1], pars.action_size);
    action_bound = pars.action_bound;

    register_module("lstm", lstm);
    register_module("lkrelu", lkrelu);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
  }
  torch::Tensor forward(torch::Tensor x) {
    x = std::get<0>(lstm->forward(x));
    x = x.transpose(0, 1)[-1];
    x = lkrelu(x);
    x = lkrelu(fc1(x));
    x = lkrelu(fc2(x));
    x = torch::tanh(fc3(x));
    return x * action_bound;
  }

private:
  torch::nn::LSTM lstm = nullptr;
  torch::nn::LeakyReLU lkrelu = nullptr;
  torch::nn::Linear fc1 = nullptr;
  torch::nn::Linear fc2 = nullptr;
  torch::nn::Linear fc3 = nullptr;
  double action_bound;
};

class Critic : public torch::nn::Module {
public:
  Critic(Pars pars) {
    lstm = torch::nn::LSTM(
        torch::nn::LSTMOptions(pars.state_size, pars.lstm_hidden_size)
            .num_layers(pars.lstm_num_layers)
            .batch_first(true));
    lkrelu = torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(pars.leaky_relu_slope));
    fc1 = torch::nn::Linear(pars.lstm_hidden_size + pars.action_size,
                            pars.critic_hidden_units[0]);
    fc2 = torch::nn::Linear(pars.critic_hidden_units[0],
                            pars.critic_hidden_units[1]);
    fc3 = torch::nn::Linear(pars.critic_hidden_units[1], 1);

    register_module("lstm", lstm);
    register_module("lkrelu", lkrelu);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
  }
  torch::Tensor forward(torch::Tensor x, torch::Tensor a) {
    x = std::get<0>(lstm->forward(x));
    x = x.transpose(0, 1)[-1];
    x = lkrelu(x);
    x = torch::cat({x, a}, 1);
    x = lkrelu(fc1(x));
    x = lkrelu(fc2(x));
    return fc3(x);
  }

private:
  torch::nn::LSTM lstm = nullptr;
  torch::nn::LeakyReLU lkrelu = nullptr;
  torch::nn::Linear fc1 = nullptr;
  torch::nn::Linear fc2 = nullptr;
  torch::nn::Linear fc3 = nullptr;
};

#endif // __MODEL_HPP__
