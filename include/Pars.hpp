
#ifndef __PARS_HPP__
#define __PARS_HPP__

#include "Data.hpp"
#include "bits/stdc++.h"
#include "torch/torch.h"

#define LR_ACTOR 1e-4
#define LR_CRITIC 1e-3
#define GAMMA 0.99
#define TAU 0.005
#define MEMORY_SIZE 100000
#define BATCH_SIZE 64

class Pars {
public:
  Pars(int state_size = 33, int action_size = 4,
       std::vector<int> actor_hidden_units = {256, 128},
       std::vector<int> critic_hidden_units = {256, 128}, int lstm_length = 3,
       int lstm_num_layers = 1, int lstm_hidden_size = 50,
       double leaky_relu_slope = 0.1, int memory_size = MEMORY_SIZE,
       int batch_size = BATCH_SIZE, double lr_actor = LR_ACTOR,
       double lr_critic = LR_CRITIC, double gamma = GAMMA, double tau = TAU,
       double action_bound = 2, double sigma = 1,
       torch::Device device = torch::kCUDA)
      : state_size(state_size), action_size(action_size),
        actor_hidden_units(actor_hidden_units),
        critic_hidden_units(critic_hidden_units), memory_size(memory_size),
        batch_size(batch_size), lr_actor(lr_actor), lr_critic(lr_critic),
        gamma(gamma), tau(tau), action_bound(action_bound), sigma(sigma),
        device(device), lstm_length(lstm_length),
        lstm_num_layers(lstm_num_layers), lstm_hidden_size(lstm_hidden_size),
        leaky_relu_slope(leaky_relu_slope) {
    if (device == torch::kCUDA && !torch::cuda::is_available()) {
      device = torch::kCPU;
    }
  }

  Pars(const Pars &pars)
      : state_size(pars.state_size), action_size(pars.action_size),
        actor_hidden_units(pars.actor_hidden_units),
        critic_hidden_units(pars.critic_hidden_units),
        memory_size(pars.memory_size), batch_size(pars.batch_size),
        lr_actor(pars.lr_actor), lr_critic(pars.lr_critic), gamma(pars.gamma),
        tau(pars.tau), action_bound(pars.action_bound), sigma(pars.sigma),
        device(pars.device), // Initialize device explicitly
        lstm_length(pars.lstm_length), lstm_num_layers(pars.lstm_num_layers),
        lstm_hidden_size(pars.lstm_hidden_size),
        leaky_relu_slope(pars.leaky_relu_slope) {
    lstm_num_layers = pars.lstm_num_layers;
    lstm_hidden_size = pars.lstm_hidden_size;
    leaky_relu_slope = pars.leaky_relu_slope;
  }

  void adp(Data &data) {
    state_size = data.state_seq.size(-1);
    action_size = data.action.size(-1);
    device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  }

  int state_size;
  int action_size;
  std::vector<int> actor_hidden_units;
  std::vector<int> critic_hidden_units;
  int memory_size;
  int batch_size;
  double lr_actor;
  double lr_critic;
  double gamma;
  double tau;
  double action_bound;
  double sigma;
  torch::Device device;
  int lstm_length;
  int lstm_num_layers;
  int lstm_hidden_size;
  double leaky_relu_slope;
};

#endif //__PARS_HPP__
