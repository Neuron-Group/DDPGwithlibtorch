
#pragma once

#ifndef __AGENT_HPP__
#define __AGENT_HPP__

#include "Data.hpp"
#include "Model.hpp"
#include "Noise.hpp"
#include "Pars.hpp"
#include "ReplayMemory.hpp"
#include "bits/stdc++.h"
#include "torch/script.h"
#include "torch/serialize.h"
#include "torch/serialize/tensor.h"
#include "torch/torch.h"

class Agent {
public:
  Actor *actor;
  Actor *actor_target;
  torch::optim::Adam *actor_optimizer;
  Critic *critic;
  Critic *critic_target;
  torch::optim::Adam *critic_optimizer;
  ReplayMemory *replay_buffer;
  std::deque<torch::Tensor> *state_buffer;
  std::deque<torch::Tensor> *state_buffer_next;
  Pars *pars;
  Noise *noise;

  Agent(Pars pars) {
    /*
    self.actor = Actor(pars).to(pars.device);  # move nn to device
    self.actor_target = Actor(pars).to(pars.device);  # same structure as actor
    self.actor_target.load_state_dict(self.actor.state_dict());  # copy the
    current nn's weights of actor self.actor_optimizer =
    optim.Adam(self.actor.parameters(), lr=pars.lr_actor);  # retrieves the
    parameters
    */
    this->actor = new Actor(pars);
    this->actor_target = new Actor(pars);
    actor->to(pars.device);
    actor->train();
    actor_target->to(pars.device);
    actor_target->train();
    for (size_t i = 0; i < this->actor->parameters().size(); i++) {
      this->actor_target->parameters()[i].data().copy_(
          this->actor->parameters()[i].data());
    }
    this->actor_optimizer = new torch::optim::Adam(
        this->actor->parameters(), torch::optim::AdamOptions(pars.lr_actor));

    /*
    self.critic = Critic(pars).to(pars.device);
    self.critic_target = Critic(pars).to(pars.device);
    self.critic_target.load_state_dict(self.critic.state_dict());
    self.critic_optimizer = optim.Adam(self.critic.parameters(),
    lr=pars.lr_critic);
    */
    this->critic = new Critic(pars);
    this->critic_target = new Critic(pars);
    critic->to(pars.device);
    critic->train();
    critic_target->to(pars.device);
    critic_target->train();
    for (size_t i = 0; i < this->critic->parameters().size(); i++) {
      this->critic_target->parameters()[i].data().copy_(
          this->critic->parameters()[i].data());
    }
    this->critic_optimizer = new torch::optim::Adam(
        this->critic->parameters(), torch::optim::AdamOptions(pars.lr_critic));

    /*
    self.replay_buffer = ReplayMemory(pars);  # create a replay buffer
    self.state_buffer = deque(maxlen=pars.lstm_length);
    self.state_buffer_next = deque(maxlen=pars.lstm_length);
    */
    this->replay_buffer = new ReplayMemory(pars);
    this->state_buffer = new std::deque<torch::Tensor>;
    this->state_buffer_next = new std::deque<torch::Tensor>;

    this->noise = new Noise();

    this->pars = new Pars(pars);
  }
  /*
       def get_action(self, state, step_i):
       # state = torch.FloatTensor(state).unsqueeze(0).to(self.pars.device);  #
     unsqueeze(0) add a dimension from (3,) to (1,3)
       self.state_buffer.append(state);
       state_seq = deque_to_tensor(self.state_buffer,
     self.pars).unsqueeze(0).to(self.pars.device); action =
     self.actor(state_seq); action = action.detach().cpu().numpy()[0];  # detach
     the tensor from the current graph and convert it to numpy

       return Noise.gauss(action, 0.8**step_i , self.pars);
  */
  torch::Tensor get_action(torch::Tensor state, size_t step_i) {
    this->state_buffer->push_back(state);
    this->replay_buffer->resize_front_back(this->state_buffer,
                                           this->pars->lstm_length);
    torch::Tensor state_seq =
        torch::stack(std::vector<torch::Tensor>{this->state_buffer->begin(),
                                                this->state_buffer->end()})
            .clone()
            .unsqueeze(0)
            .to(torch::kFloat32)
            .to(this->pars->device);
    torch::Tensor action = this->actor->forward(state_seq);
    action = action.detach().cpu().clone();
    return this->noise->gauss(action, pow(0.8, step_i), *this->pars);
  }

  void update() {
    /*
    if len(self.replay_buffer) < self.pars.batch_size:
         return  # skip the update if the replay buffer is not filled enough

    # states, actions, rewards, next_states, dones =
    self.replay_buffer.sample(batch_size) #
    首先采样出作为Data类的对象，然后分别取出state, action, reward, next_state,
    done samples = self.replay_buffer.sample(self.pars); states_seq_sample =
    [sample.state_seq for sample in samples];
    */

    if (this->replay_buffer->__len__() < this->pars->batch_size) {
      return;
    }

    std::vector<Data> samples = this->replay_buffer->sample();
    std::vector<torch::Tensor> states_seq_sample_vector;

    for (size_t i = 0; i < samples.size(); i++) {
      states_seq_sample_vector.push_back(samples[i].state_seq);
    }

    /*
    # 将上面形式的数据转换为tensor
    for i in range(len(states_seq_sample)):
         states_seq_sample[i] = deque_to_tensor(states_seq_sample[i],
    self.pars); states_seq_sample = torch.stack(states_seq_sample, 0);
    states_seq_sample = states_seq_sample.float().to(self.pars.device);
    */
    torch::Tensor states_seq_sample = torch::stack(states_seq_sample_vector, 0);
    states_seq_sample =
        states_seq_sample.to(torch::kFloat32).to(this->pars->device);

    /*
    actions = [sample.action for sample in samples];
    rewards = [sample.reward for sample in samples];
    next_states_seq_sample = [sample.next_state_seq for sample in samples];
    dones = [sample.done for sample in samples];
    */
    std::vector<torch::Tensor> actions_vector;
    std::vector<torch::Tensor> rewards_vector;
    std::vector<torch::Tensor> next_states_seq_sample_vector;
    std::vector<torch::Tensor> dones_vector;

    for (size_t i = 0; i < samples.size(); i++) {
      actions_vector.push_back(samples[i].action);
      rewards_vector.push_back(samples[i].reward);
      next_states_seq_sample_vector.push_back(samples[i].next_state_seq);
      dones_vector.push_back(samples[i].done);
    }

    /*
    actions = torch.FloatTensor(np.vstack(actions)).to(self.pars.device);
    # actions = torch.FloatTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.pars.device);
    # next_states = torch.FloatTensor(next_states).to(self.pars.device);
    for i in range(len(next_states_seq_sample)):
         next_states_seq_sample[i] = deque_to_tensor(next_states_seq_sample[i],
    self.pars); next_states_seq_sample = torch.stack(next_states_seq_sample, 0);
    next_states_seq_sample =
    next_states_seq_sample.float().to(self.pars.device);

    dones = torch.FloatTensor(dones).unsqueeze(1).to(self.pars.device);
    */
    torch::Tensor actions = torch::stack(actions_vector, 0);
    actions = actions.to(torch::kFloat32).to(this->pars->device);
    torch::Tensor rewards = torch::stack(rewards_vector, 0).unsqueeze(1);
    rewards = rewards.to(torch::kFloat32).to(this->pars->device);
    torch::Tensor next_states_seq_sample =
        torch::stack(next_states_seq_sample_vector, 0);
    next_states_seq_sample =
        next_states_seq_sample.to(torch::kFloat32).to(this->pars->device);
    torch::Tensor dones = torch::stack(dones_vector, 0).unsqueeze(1);
    dones = dones.to(torch::kFloat32).to(this->pars->device);

    /*
    # Critic update
    next_actions = self.actor_target(next_states_seq_sample);
    # .detach() means the gradient won't be back propagate to the actor
    target_Q = self.critic_target(next_states_seq_sample,
    next_actions.detach()); target_Q = rewards + (self.pars.gamma * target_Q *
    (1 - dones)); current_Q = self.critic(states_seq_sample, actions);
    */
    torch::Tensor next_actions =
        this->actor_target->forward(next_states_seq_sample);
    torch::Tensor target_Q = this->critic_target->forward(
        next_states_seq_sample, next_actions.detach());
    target_Q = rewards + (this->pars->gamma * target_Q * (1 - dones));
    torch::Tensor current_Q = this->critic->forward(states_seq_sample, actions);
    /*
    # nn.MSELoss() means Mean Squared Error
    critic_loss = nn.MSELoss()(current_Q, target_Q.detach());
    self.critic_optimizer.zero_grad();
    critic_loss.backward();
    self.critic_optimizer.step();
    */
    torch::Tensor critic_loss =
        torch::nn::MSELoss()(current_Q, target_Q.detach());
    this->critic_optimizer->zero_grad();
    critic_loss.backward();
    this->critic_optimizer->step();

    /*
    # Actor update
    actor_loss = -self.critic(states_seq_sample,
    self.actor(states_seq_sample)).mean();

    # .mean() is to calculate the mean of the tensor
    self.actor_optimizer.zero_grad();
    actor_loss.backward();
    self.actor_optimizer.step();
    */
    torch::Tensor actor_loss =
        -(this->critic
              ->forward(states_seq_sample,
                        this->actor->forward(states_seq_sample))
              .mean());
    this->actor_optimizer->zero_grad();
    actor_loss.backward();
    this->actor_optimizer->step();

    /*
    # Update target networks
    for target_param, param in zip(self.actor_target.parameters(),
    self.actor.parameters()): target_param.data.copy_(self.pars.tau * param.data
    + (1.0 - self.pars.tau) * target_param.data)

    for target_param, param in zip(self.critic_target.parameters(),
    self.critic.parameters()): target_param.data.copy_(self.pars.tau *
    param.data + (1.0 - self.pars.tau) * target_param.data)
    */
    /*
    for (size_t i = 0; i < this->actor->parameters().size(); i++) {
         this->actor_target->parameters()[i].data().copy_(
              this->actor->parameters()[i].data());
    }
    */
    for (size_t i = 0; i < this->actor->parameters().size(); i++) {
      this->actor_target->parameters()[i].data().copy_(
          this->pars->tau * this->actor->parameters()[i].data() +
          (1.0 - this->pars->tau) * this->actor_target->parameters()[i]);
    }
    for (size_t i = 0; i < this->critic->parameters().size(); i++) {
      this->critic_target->parameters()[i].data().copy_(
          this->pars->tau * this->critic->parameters()[i].data() +
          (1.0 - this->pars->tau) * this->critic_target->parameters()[i]);
    }
  }

  // 向 ReplayMemory 中添加数据

  void add_memo(torch::Tensor state, torch::Tensor action, torch::Tensor reward,
                torch::Tensor done) {
    this->replay_buffer->add_memo(state, action, reward, done);
  }

  // 清空 ReplayMemory
  void memory_clear() { this->replay_buffer->clear(); }

  void save() {
    torch::serialize::OutputArchive output_archive_1;
    this->actor->save(output_archive_1);
    output_archive_1.save_to("../model/actor.pt");

    torch::serialize::OutputArchive output_archive_2;
    this->actor_target->save(output_archive_2);
    output_archive_2.save_to("../model/actor_target.pt");

    torch::serialize::OutputArchive output_archive_3;
    this->critic->save(output_archive_3);
    output_archive_3.save_to("../model/critic.pt");

    torch::serialize::OutputArchive output_archive_4;
    this->critic_target->save(output_archive_4);
    output_archive_4.save_to("../model/critic_target.pt");
  }

  void load() {
    torch::serialize::InputArchive input_archive_1;
    input_archive_1.load_from("../model/actor.pt");
    this->actor->load(input_archive_1);

    torch::serialize::InputArchive input_archive_2;
    input_archive_2.load_from("../model/actor_target.pt");
    this->actor_target->load(input_archive_2);

    torch::serialize::InputArchive input_archive_3;
    input_archive_3.load_from("../model/critic.pt");
    this->critic->load(input_archive_3);

    torch::serialize::InputArchive input_archive_4;
    input_archive_4.load_from("../model/critic_target.pt");
    this->critic_target->load(input_archive_4);
  }
};

#endif // __AGENT_HPP__
