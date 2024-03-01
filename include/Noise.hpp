
#ifndef __NOISE_HPP__
#define __NOISE_HPP__

#include "Pars.hpp"
#include "bits/stdc++.h"
#include "torch/torch.h"

/*
class Noise:
    def __init__(self):
        return;
    def gauss(action, epsilon, pars:Pars):
        return action + epsilon * pars.sigma *
np.random.randn(pars.action_size); def conditional(action, epsilon, pars:Pars):
        if random.random() < epsilon:
            return np.random.uniform(-pars.action_bound, pars.action_bound,
pars.action_size); else: return action;
*/

class Noise {
public:
  Noise() { return; }
  torch::Tensor gauss(torch::Tensor & action, double epsilon, Pars & pars) {
    return action + epsilon * pars.sigma * torch::randn({pars.action_size});
  }
  torch::Tensor conditional(torch::Tensor action, double epsilon, Pars pars) {
    if (torch::rand({1}).item<double>() < epsilon) {
      return torch::rand({pars.action_size}) * 2 * pars.action_bound -
             pars.action_bound;
    } else {
      return action;
    }
  }
};

#endif // __NOISE_HPP__
