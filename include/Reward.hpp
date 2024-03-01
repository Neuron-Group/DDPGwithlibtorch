
#pragma once
#ifndef __REWARD_HPP__
#define __REWARD_HPP__

#include "Data.hpp"
#include "Pars.hpp"
#include "ReplayMemory.hpp"
#include "bits/stdc++.h"
#include "torch/torch.h"

// [state_seq, action] -> [state_next_predict]
// reward = -mse(state_next_predict, state_next)
class Reward_NetImpl : public torch::nn::Module {
public:
  Reward_NetImpl(Pars pars) {
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
    fc3 = torch::nn::Linear(pars.critic_hidden_units[1], pars.state_size);

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
TORCH_MODULE(Reward_Net);

class Reward {
public:
  Reward_NetImpl *net;
  torch::optim::Adam *optimizer;
  Pars *pars;
  bool is_train;
  ReplayMemory *memory;

  torch::Tensor action_;
  // torch::Tensor reward_;
  torch::Tensor done_;

  // 内置状态队列
  std::deque<torch::Tensor> *state_seq;

  Reward(Pars pars) {
    this->is_train = true;
    this->net = new Reward_NetImpl(pars);
    this->optimizer = new torch::optim::Adam(net->parameters(), pars.lr_critic);
    this->pars = new Pars(pars);
    this->memory =
        new ReplayMemory(pars); // Remove the asterisk (*) operator here
    this->state_seq = new std::deque<torch::Tensor>;
  }

  void train(ReplayMemory *memory) {
    if (this->is_train == false) {
      return;
    }
    this->net->train();
    if (memory->__len__() < pars->batch_size) {
      return;
    }

    std::vector<Data> data = memory->sample();
    std::vector<torch::Tensor> state_seq_vector;
    std::vector<torch::Tensor> action_vector;
    std::vector<torch::Tensor> next_state_vector;
    for (int i = 0; i < pars->batch_size; i++) {
      state_seq_vector.push_back(data[i].state_seq);
      action_vector.push_back(data[i].action);

      // next_state_seq 取出最后一个切片放入 next_state_vector
      next_state_vector.push_back(data[i].next_state_seq[-1].clone());
    }
    torch::Tensor state_seq_sample = torch::stack(state_seq_vector, 0);
    state_seq_sample =
        state_seq_sample.to(torch::kFloat32).to(this->pars->device);
    torch::Tensor action_sample = torch::stack(action_vector, 0);
    action_sample = action_sample.to(torch::kFloat32).to(this->pars->device);
    torch::Tensor next_state_sample = torch::stack(next_state_vector, 0);
    next_state_sample =
        next_state_sample.to(torch::kFloat32).to(this->pars->device);
    // next_state_sample 现在的维度应该是 (batch_size, state_size)
    optimizer->zero_grad();
    torch::Tensor state_next_predict =
        net->forward(state_seq_sample, action_sample);
    torch::Tensor loss =
        torch::nn::MSELoss()(state_next_predict, next_state_sample);
    loss.backward();
    optimizer->step();
  }

  void state_train() {
    this->is_train = true;
    this->net->train();
  }

  void state_eval() {
    this->is_train = false;
    this->net->eval();
  }

  torch::Tensor get_reward(torch::Tensor state, torch::Tensor action,
                           torch::Tensor done) {
    if (this->is_train) {
      this->memory->add_memo(state, action, torch::zeros({1}), done);
      this->train(this->memory);
    }
    // 如果状态队列小于 lstm_length，将 state 放入队列，返回 0
    if (this->state_seq->size() < this->pars->lstm_length) {
      this->state_seq->push_back(state.clone());
      this->action_ = action.clone();
      // this->reward_ = torch::zeros({1});
      this->done_ = done.clone();
      return torch::zeros({1});
    } else {
      // 如果状态队列大于等于 lstm_length
      // 先将队列整形为 lstm_length
      this->memory->resize_front(this->state_seq, this->pars->lstm_length);
      // 将队列转换为 tensor
      torch::Tensor state_seq =
          torch::stack(std::vector<torch::Tensor>{this->state_seq->begin(),
                                                  this->state_seq->end()})
              .clone()
              .to(torch::kFloat32)
              .to(this->pars->device);
      // 将 state 放入队列
      this->state_seq->push_back(state.clone());
      // 将队列整形为 lstm_length
      this->memory->resize_front(this->state_seq, this->pars->lstm_length);
      // 利用网络进行预测
      torch::Tensor reward = -torch::nn::MSELoss()(
          this->net->forward(state_seq, this->action_), state);
      this->action_ = action.clone();
      // this->reward_ = reward.clone();
      this->done_ = done.clone();
      return reward.detach().clone();
    }
    return torch::zeros({1});
  }

  // 清空 ReplayMemory
  void memory_clear() { this->memory->clear(); }

  // 保存模型
  // 路径为 ../model/reward_net.pt
  void save() {
    torch::serialize::OutputArchive output_archive;
    this->net->save(output_archive);
    output_archive.save_to("../model/reward_net.pt");
  }

  // 加载模型
  // 路径为 ../model/reward_net.pt
  void load() {
    torch::serialize::InputArchive input_archive;
    input_archive.load_from("../model/reward_net.pt");
    this->net->load(input_archive);
  }
};

#endif // __REWARD_HPP__
