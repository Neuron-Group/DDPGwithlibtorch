
#ifndef __DATA_HPP__
#define __DATA_HPP__

#include "bits/stdc++.h"
#include "torch/torch.h"

class Data {
public:
  torch::Tensor state_seq;
  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor next_state_seq;
  torch::Tensor done;

  Data(torch::Tensor state_seq, torch::Tensor action, torch::Tensor reward,
       torch::Tensor next_state_seq, torch::Tensor done) {
    this->state_seq = state_seq.clone();
    this->action = action.clone();
    this->reward = reward.clone();
    this->next_state_seq = next_state_seq.clone();
    this->done = done.clone();
  }
};

#endif // __DATA_HPP__
