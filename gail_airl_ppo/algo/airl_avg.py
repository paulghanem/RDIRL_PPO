import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from .avg import AVG
from gail_airl_ppo.network import AIRLDiscrim


class AIRL_AVG(AVG):
    """AIRL with AVG as the inner policy optimization algorithm"""

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.99, buffer_size=100000, batch_size=1,
                 lr_actor=0.0063, lr_critic=0.0087, lr_disc=3e-5,
                 units_actor=(256, 256), units_critic=(256, 256),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 alpha=0.07, beta1=0.0, max_grad_norm=10.0,
                 disc_update_freq=1, name=None):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, buffer_size,
            lr_actor, lr_critic, units_actor, units_critic,
            alpha, beta1, max_grad_norm
        )
        self.name = name

        # Expert's buffer
        self.buffer_exp = buffer_exp

        # Discriminator (AIRL has separate reward and value networks)
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=False),
            hidden_activation_v=nn.ReLU(inplace=False)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = SGD(self.disc.parameters(), lr=lr_disc, momentum=0.9)
        self.batch_size = batch_size
        self.disc_update_freq = disc_update_freq

    def is_update(self, step, algo=None):
        """AVG updates every step"""
        return True

    def update(self, writer=None, step=None):
        """Update discriminator and AVG policy"""
        self.learning_steps += 1

        # Update discriminator periodically
        if self.learning_steps % self.disc_update_freq == 0 and self.buffer._n >= self.batch_size:
            self.learning_steps_disc += 1

            # Get samples from current policy
            if step is not None and self.buffer._n >= step:
                idx = (step - 1) % self.buffer._n
                states, _, _, dones, log_pis, next_states = \
                    self.buffer.get_sample(idx)
            else:
                states, _, _, dones, log_pis, next_states = \
                    self.buffer.sample(self.batch_size)

            # Get samples from expert
            if step is not None:
                idx_exp = (step - 1) % self.buffer_exp.buffer_size
                states_exp, actions_exp, _, dones_exp, next_states_exp = \
                    self.buffer_exp.get_sample(idx_exp)
            else:
                states_exp, actions_exp, _, dones_exp, next_states_exp = \
                    self.buffer_exp.sample(self.batch_size)

            # Calculate log probabilities of expert actions
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)

            # Update discriminator
            self.update_disc(
                states, dones, log_pis, next_states,
                states_exp, dones_exp, log_pis_exp, next_states_exp,
                writer
            )

        # Get the most recent transition for AVG update
        if self.buffer._n >= 1:
            states, actions, _, dones, log_pis, next_states = \
                self.buffer.sample(1)

            # Ensure proper shapes
            state = states[0] if states.dim() > 1 else states
            action = actions[0] if actions.dim() > 1 else actions
            done = dones[0].item() if dones.dim() > 0 else dones.item()
            log_pi = log_pis[0].item() if log_pis.dim() > 0 else log_pis.item()
            next_state = next_states[0] if next_states.dim() > 1 else next_states

            # Calculate reward from AIRL discriminator
            with torch.no_grad():
                reward = self.disc.calculate_reward(
                    state.unsqueeze(0), done, log_pi, next_state.unsqueeze(0)
                ).item()

            # Update AVG using discriminator reward
            self.update_avg(state.unsqueeze(0), action.unsqueeze(0),
                          next_state.unsqueeze(0), reward, done, log_pi, writer)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp, next_states_exp,
                    writer):
        """Update AIRL discriminator"""
        # Output of discriminator is (-inf, inf), not [0, 1]
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator maximizes E_π[log(1 - D)] + E_exp[log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        # Logging
        if writer is not None and self.learning_steps_disc % 10 == 0:
            writer.add_scalar('loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator accuracies
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)

    def save_models(self, save_dir):
        """Save actor, critic, and discriminator"""
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'critic.pth'))
        torch.save(self.disc.state_dict(), os.path.join(save_dir, 'disc.pth'))


# Add method to AVGActor for AIRL compatibility
def evaluate_log_pi(self, states, actions):
    """Evaluate log probability of given actions under current policy"""
    _, _, dist = self.forward(states)
    # Inverse tanh to get pre-tanh actions
    action_pre = torch.atanh(torch.clamp(actions, -0.999, 0.999))
    lprob = dist.log_prob(action_pre)
    # Correction for tanh transformation
    lprob -= (2 * (torch.log(torch.tensor(2.0)) - action_pre -
              F.softplus(-2 * action_pre))).sum(axis=1)
    return lprob


# Monkey patch the method into AVGActor
from .avg import AVGActor
AVGActor.evaluate_log_pi = evaluate_log_pi
