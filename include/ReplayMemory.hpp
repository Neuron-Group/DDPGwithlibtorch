
#pragma once

#ifndef __REPLAYMEMORY_HPP__
#define __REPLAYMEMORY_HPP__

#include "bits/stdc++.h"
#include "torch/torch.h"

#include "Data.hpp"
#include "Pars.hpp"

class ReplayMemory {
private:
  torch::Tensor action_;
  torch::Tensor reward_;
  torch::Tensor done_;

public:
  std::deque<Data> *buffer;       // 存放 replay buffer 中的所有 data
  std::deque<torch::Tensor> *seq; // 仅仅存放一个序列的 state
  Pars pars;

  void resize_front(std::deque<torch::Tensor> *tensor_que, int size) {
    while (tensor_que->size() > size) {
      tensor_que->pop_front();
    }
  }

  void resize_front(std::deque<Data> *data_que, int size) {
    while (data_que->size() > size) {
      data_que->pop_front();
    }
  }

  void resize_front_back(std::deque<torch::Tensor> *tensor_que, int size) {
    while (tensor_que->size() > size) {
      tensor_que->pop_front();
    }
    // 如果队列中没有元素
    // 直接返回
    if (tensor_que->size() == 0) {
      return;
    }
    while (tensor_que->size() < size) {
      // 用最早放入的元素在尾部填充队列
      tensor_que->push_front(tensor_que->front());
    }
  }

  ReplayMemory(Pars pars) : pars(pars) {
    buffer = new std::deque<Data>;
    seq = new std::deque<torch::Tensor>;
  }

  /*
  void add_memo(torch::Tensor action, torch::Tensor reward,
                torch::Tensor next_state, torch::Tensor done) {
    if (state_next_flag) {
      if (this->seq->size() == 0) {
        // 如果队列中没有元素，先更新 action, reward, done
        this->action_ = action.clone();
        this->reward_ = reward.clone();
        this->done_ = done.clone();
        return;
      }

      torch::Tensor seq_state;
      bool seq_state_flag = false;
      torch::Tensor seq_next_state;

      if (seq->size() >= pars.lstm_length) {
        resize_front(seq, pars.lstm_length);
        // 如果队列长度达到pars.lstm_length，将队列中的状态转换为一个tensor
        // 并赋值给 seq_state
        // 将seq_state_flag置为true
        seq_state =
            torch::stack(std::vector<torch::Tensor>{seq->begin(), seq->end()})
                .clone()
                .to(torch::kFloat16);
        seq_state_flag = true;
      }

      // 向队列中添加第二个状态
      seq->push_back(next_state.clone());
      // 如果队列长度达到pars.lstm_length，将队列中的状态转换为一个tensor
      // 并赋值给 seq_next_state
      if (seq->size() >= pars.lstm_length) {
        // 踢出队尾的状态，保持队列长度为pars.lstm_length
        resize_front(seq, pars.lstm_length);
        seq_next_state =
            torch::stack(std::vector<torch::Tensor>{seq->begin(), seq->end()})
                .clone()
                .to(torch::kFloat16);
      }

      // 如果seq_state_flag为true，将seq_state和seq_next_state添加到buffer中
      // action, reward, done 要用上一次的
      if (seq_state_flag) {
        buffer->push_back(Data(seq_state, this->action_, this->reward_,
                               seq_next_state, this->done_));
      }
      // 此时 seq_state 和 seq_next_state 的维度应该为 (pars.lstm_length,
      // state_size)

      // 更新 action, reward, done
      this->action_ = action.clone();
      this->reward_ = reward.clone();
      this->done_ = done.clone();
    } else {
      torch::Tensor seq_state;
      bool seq_state_flag = false;
      torch::Tensor seq_next_state;

      if (seq->size() >= pars.lstm_length) {
        resize_front(seq, pars.lstm_length);
        // 如果队列长度达到pars.lstm_length，将队列中的状态转换为一个tensor
        // 并赋值给 seq_state
        // 将seq_state_flag置为true
        seq_state =
            torch::stack(std::vector<torch::Tensor>{seq->begin(), seq->end()})
                .clone()
                .to(torch::kFloat16);
        seq_state_flag = true;
      }

      // 向队列中添加第二个状态
      seq->push_back(next_state.clone());
      // 如果队列长度达到pars.lstm_length，将队列中的状态转换为一个tensor
      // 并赋值给 seq_next_state
      if (seq->size() >= pars.lstm_length) {
        // 踢出队尾的状态，保持队列长度为pars.lstm_length
        resize_front(seq, pars.lstm_length);
        seq_next_state =
            torch::stack(std::vector<torch::Tensor>{seq->begin(), seq->end()})
                .clone()
                .to(torch::kFloat16);
      }

      // 如果seq_state_flag为true，将seq_state和seq_next_state添加到buffer中
      // action, reward, done 用这一次的
      if (seq_state_flag) {
        buffer->push_back(
            Data(seq_state, action, reward, seq_next_state, done));
      }
      // 此时 seq_state 和 seq_next_state 的维度应该为 (pars.lstm_length,
      // state_size)
    }
    this->resize_front(this->buffer, pars.memory_size);
  }
  */
  void add_memo(torch::Tensor state, torch::Tensor action, torch::Tensor reward,
                torch::Tensor done) {
    if (this->seq->size() == 0) {
      // 如果队列中没有元素，先更新 action, reward, done
      this->action_ = action.clone();
      this->reward_ = reward.clone();
      this->done_ = done.clone();
      return;
    }

    torch::Tensor seq_state;
    bool seq_state_flag = false;
    torch::Tensor seq_next_state;

    if (seq->size() >= pars.lstm_length) {
      resize_front(seq, pars.lstm_length);
      // 如果队列长度达到pars.lstm_length，将队列中的状态转换为一个tensor
      // 并赋值给 seq_state
      // 将seq_state_flag置为true
      seq_state =
          torch::stack(std::vector<torch::Tensor>{seq->begin(), seq->end()})
              .clone()
              .to(torch::kFloat16);
      seq_state_flag = true;
    }

    // 向队列中添加第二个状态
    seq->push_back(state.clone());
    // 如果队列长度达到pars.lstm_length，将队列中的状态转换为一个tensor
    // 并赋值给 seq_next_state
    if (seq->size() >= pars.lstm_length) {
      // 踢出队尾的状态，保持队列长度为pars.lstm_length
      resize_front(seq, pars.lstm_length);
      seq_next_state =
          torch::stack(std::vector<torch::Tensor>{seq->begin(), seq->end()})
              .clone()
              .to(torch::kFloat16);
    }

    // 如果seq_state_flag为true，将seq_state和seq_next_state添加到buffer中
    // action, reward, done 要用上一次的
    if (seq_state_flag) {
      buffer->push_back(Data(seq_state, this->action_, this->reward_,
                             seq_next_state, this->done_));
    }
    // 此时 seq_state 和 seq_next_state 的维度应该为 (pars.lstm_length,
    // state_size)

    // 更新 action, reward, done
    this->action_ = action.clone();
    this->reward_ = reward.clone();
    this->done_ = done.clone();

    this->resize_front(this->buffer, pars.memory_size);
  }

  std::vector<Data> sample() {
    std::vector<Data> result;
    std::sample(buffer->begin(), buffer->end(), std::back_inserter(result),
                pars.batch_size, std::mt19937{std::random_device{}()});
    return result;
  }

  // 清空队列，恢复初始状态
  void clear() {
    this->buffer->clear();
    this->seq->clear();
  }

  int __len__() { return buffer->size(); }
};

#endif //__REPLAYMEMORY_HPP__
