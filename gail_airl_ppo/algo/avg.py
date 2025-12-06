import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .base import Algorithm
from gail_airl_ppo.buffer import Buffer
from gail_airl_ppo.network import StateFunction


def orthogonal_weight_init(m):
    """Orthogonal weight initialization for neural networks"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class TDErrorScaler:
    """Scales TD errors to have unit variance for stable learning"""
    def __init__(self):
        self.sigma = 1.0
        self.alpha = 0.01

    def update(self, reward, gamma, G=None):
        if G is not None:
            # Episode ended, update with actual return
            td_error = abs(G)
        else:
            # Intermediate step, use reward as proxy
            td_error = abs(reward)

        self.sigma = (1 - self.alpha) * self.sigma + self.alpha * (td_error + 1e-8)


class AVGActor(nn.Module):
    """Continuous MLP Actor for AVG"""
    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.LeakyReLU(), device='cpu'):
        super().__init__()
        self.device = device
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # Build hidden layers
        layers = []
        in_dim = state_shape[0]
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                hidden_activation
            ])
            in_dim = hidden_dim

        self.phi = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, action_shape[0])
        self.log_std = nn.Linear(in_dim, action_shape[0])

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs):
        phi = self.phi(obs.to(self.device))
        phi = phi / (torch.norm(phi, dim=1, keepdim=True) + 1e-8)
        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)

        # Tanh applied after lprob calculation
        action = torch.tanh(action_pre)

        return action, lprob, dist

    def sample(self, obs):
        """For compatibility with base Algorithm class"""
        action, lprob, _ = self.forward(obs)
        return action, lprob


class AVGQ(nn.Module):
    """Q-function for AVG"""
    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.LeakyReLU(), device='cpu'):
        super().__init__()
        self.device = device

        # Build hidden layers
        layers = []
        in_dim = state_shape[0] + action_shape[0]
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                hidden_activation
            ])
            in_dim = hidden_dim

        self.phi = nn.Sequential(*layers)
        self.q = nn.Linear(in_dim, 1)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1).to(self.device)
        phi = self.phi(x)
        phi = phi / (torch.norm(phi, dim=1, keepdim=True) + 1e-8)
        return self.q(phi).squeeze(-1)


class AVG(Algorithm):
    """Action Value Gradient algorithm adapted for the imitation learning framework"""

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.99,
                 buffer_size=100000, lr_actor=0.0063, lr_critic=0.0087,
                 units_actor=(256, 256), units_critic=(256, 256),
                 alpha=0.07, beta1=0.0, max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Experience buffer (for compatibility, though AVG is online)
        self.buffer = Buffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        # Actor
        self.actor = AVGActor(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.LeakyReLU(),
            device=device
        )

        # Critic (Q-function)
        self.critic = AVGQ(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.LeakyReLU(),
            device=device
        )

        # Optimizers
        betas = (beta1, 0.999)
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor, betas=betas)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic, betas=betas)

        # AVG-specific parameters
        self.alpha = alpha  # Entropy coefficient
        self.max_grad_norm = max_grad_norm
        self.td_error_scaler = TDErrorScaler()
        self.G = 0  # Cumulative return for current episode

        self.learning_steps_avg = 0
        self.last_log_pi = 0.0  # Store most recent log_pi

    def is_update(self, step, algo=None):
        """AVG updates every step (online learning)"""
        return True

    def exploit(self, state):
        """Get action for evaluation (deterministic)"""
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, _, _ = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    def step(self, env, state, t, step):
        """Collect a single transition"""
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        mask = False if t == env._max_episode_steps else done

        # Store log_pi for use in update
        self.last_log_pi = log_pi

        # Store in buffer (Buffer doesn't store log_pi)
        self.buffer.append(state, action, reward, mask, next_state)

        if done:
            t = 0
            next_state, _ = env.reset()

        return next_state, t

    def update(self, step=None):
        """Update actor and critic using AVG"""
        self.learning_steps += 1
        self.learning_steps_avg += 1

        # Get the most recent transition
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(1)

        # Ensure proper shapes
        state = states[0] if states.dim() > 1 else states
        action = actions[0] if actions.dim() > 1 else actions
        reward = rewards[0].item() if rewards.dim() > 0 else rewards.item()
        done = dones[0].item() if dones.dim() > 0 else dones.item()
        next_state = next_states[0] if next_states.dim() > 1 else next_states

        # Use the log_pi from when action was taken
        log_pi = self.last_log_pi

        self.update_avg(state.unsqueeze(0), action.unsqueeze(0),
                       next_state.unsqueeze(0), reward, done, log_pi, None)

    def update_avg(self, state, action, next_state, reward, done, log_pi, writer):
        """Core AVG update"""
        # Return scaling
        r_ent = reward - self.alpha * log_pi
        self.G += r_ent
        if done:
            self.td_error_scaler.update(reward=r_ent, gamma=0, G=self.G)
            self.G = 0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.gamma, G=None)

        # Q loss
        q = self.critic(state, action.detach())  # No gradient through action
        with torch.no_grad():
            next_action, next_lprob, _ = self.actor(next_state)
            q_next = self.critic(next_state, next_action)
            target_V = q_next - self.alpha * next_lprob

        delta = reward + (1 - done) * self.gamma * target_V - q
        delta /= (self.td_error_scaler.sigma + 1e-8)
        loss_critic = delta ** 2

        # Update critic
        self.optim_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        # Policy loss
        new_action, new_lprob, _ = self.actor(state)
        loss_actor = self.alpha * new_lprob - self.critic(state, new_action)

        # Update actor
        self.optim_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        # Logging (only if writer is provided and not None)
        if writer is not None and hasattr(writer, 'add_scalar') and self.learning_steps_avg % 100 == 0:
            writer.add_scalar('loss/critic', loss_critic.item(), self.learning_steps)
            writer.add_scalar('loss/actor', loss_actor.item(), self.learning_steps)

    def save_models(self, save_dir):
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'critic.pth'))
