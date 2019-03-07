import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO(object):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, a1_rollouts, a2_rollouts):
        a1_advantages = a1_rollouts.returns[:-1] - a1_rollouts.value_preds[:-1]
        a1_advantages = (a1_advantages - a1_advantages.mean()) / (
            a1_advantages.std() + 1e-5)

        a2_advantages = a2_rollouts.returns[:-1] - a2_rollouts.value_preds[:-1]
        a2_advantages = (a2_advantages - a2_advantages.mean()) / (
            a2_advantages.std() + 1e-5)


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if hasattr(self.actor_critic.base, 'gru'):
                a1_data_generator = a1_rollouts.recurrent_generator(
                    a1_advantages, self.num_mini_batch)
                a2_data_generator = a2_rollouts.recurrent_generator(
                    a2_advantages, self.num_mini_batch)
            else:
                a1_data_generator = a1_rollouts.feed_forward_generator(
                    a1_advantages, self.num_mini_batch)
                a2_data_generator = a2_rollouts.feed_forward_generator(
                    a2_advantages, self.num_mini_batch)

            for a1_sample, a2_sample in zip(a1_data_generator, a2_data_generator):
                a1_observations_batch, a1_states_batch, a1_actions_batch, \
                   a1_return_batch, a1_masks_batch, a1_old_action_log_probs_batch, \
                        a1_adv_targ = a1_sample

                a2_observations_batch, a2_states_batch, a2_actions_batch, \
                   a2_return_batch, a2_masks_batch, a2_old_action_log_probs_batch, \
                        a2_adv_targ = a2_sample

                a2_values, a2_action_log_probs, a2_dist_entropy, a1_states, a2_states = self.actor_critic.evaluate_actions(
                    a1_observations_batch, a1_states_batch,
                    a1_masks_batch, a2_observations_batch, a2_states_batch,
                    a2_masks_batch, a2_actions_batch)

                a1_values, a1_action_log_probs, a1_dist_entropy, a1_states, a2_states = self.actor_critic.evaluate_actions(
                    a1_observations_batch, a1_states_batch,
                    a1_masks_batch, a2_observations_batch, a2_states_batch,
                    a2_masks_batch, a1_actions_batch)

                a1_ratio = torch.exp(a1_action_log_probs - a1_old_action_log_probs_batch)
                a1_surr1 = a1_ratio * a1_adv_targ
                a1_surr2 = torch.clamp(a1_ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * a1_adv_targ
                a1_action_loss = -torch.min(a1_surr1, a1_surr2).mean()
                a1_value_loss = F.mse_loss(a1_return_batch, a1_values)

                a2_ratio = torch.exp(a2_action_log_probs - a2_old_action_log_probs_batch)
                a2_surr1 = a2_ratio * a2_adv_targ
                a2_surr2 = torch.clamp(a2_ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * a2_adv_targ
                a2_action_loss = -torch.min(a2_surr1, a2_surr2).mean()
                a2_value_loss = F.mse_loss(a2_return_batch, a2_values)


                self.optimizer.zero_grad()
                (a1_value_loss * self.value_loss_coef + a1_action_loss -
                 a1_dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += a1_value_loss.item()
                action_loss_epoch += a1_action_loss.item()
                dist_entropy_epoch += a1_dist_entropy.item()

                self.optimizer.zero_grad()
                (a2_value_loss * self.value_loss_coef + a2_action_loss -
                 a2_dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += a2_value_loss.item()
                action_loss_epoch += a2_action_loss.item()
                dist_entropy_epoch += a2_dist_entropy.item()

            # for a2_sample in a2_data_generator:

            #     a2_observations_batch, a2_states_batch, a2_actions_batch, \
            #        a2_return_batch, a2_masks_batch, a2_old_action_log_probs_batch, \
            #             a2_adv_targ = a2_sample

            #     a2_values, a2_action_log_probs, a2_dist_entropy, a1_states, a2_states = self.actor_critic.evaluate_actions(
            #         a1_observations_batch, a1_states_batch,
            #         a1_masks_batch, a2_observations_batch, a2_states_batch,
            #         a2_masks_batch, a2_actions_batch)

            #     a2_ratio = torch.exp(a2_action_log_probs - a2_old_action_log_probs_batch)
            #     a2_surr1 = a2_ratio * a2_adv_targ
            #     a2_surr2 = torch.clamp(a2_ratio, 1.0 - self.clip_param,
            #                                1.0 + self.clip_param) * a2_adv_targ
            #     a2_action_loss = -torch.min(a2_surr1, a2_surr2).mean()
            #     a2_value_loss = F.mse_loss(a2_return_batch, a2_values)

            #     self.optimizer.zero_grad()
            #     (a2_value_loss * self.value_loss_coef + a2_action_loss -
            #      a2_dist_entropy * self.entropy_coef).backward()
            #     nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
            #                              self.max_grad_norm)
            #     self.optimizer.step()

            #     value_loss_epoch += a2_value_loss.item()
            #     action_loss_epoch += a2_action_loss.item()
            #     dist_entropy_epoch += a2_dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
