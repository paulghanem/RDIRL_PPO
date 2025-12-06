import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from .avg import AVG
from gail_airl_ppo.network import RGCLCost


class RGCL_AVG(AVG):
    """RGCL with AVG as the inner policy optimization algorithm

    Uses Kalman filtering for reward parameter estimation combined with
    online AVG policy updates.
    """

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.99, rollout_length=100000, buffer_size=None, batch_size=64,
                 lr_actor=0.0063, lr_critic=0.0087, lr_disc=3e-5,
                 units_actor=(256, 256), units_critic=(256, 256),
                 units_disc=(16, 16), alpha=0.07, beta1=0.0,
                 max_grad_norm=10.0, disc_update_freq=1,
                 Q_init=1e-6, P_init=1e-4, name=None):
        # Use rollout_length as buffer_size for compatibility
        if buffer_size is None:
            buffer_size = rollout_length
        super().__init__(
            state_shape, action_shape, device, seed, gamma, buffer_size,
            lr_actor, lr_critic, units_actor, units_critic,
            alpha, beta1, max_grad_norm
        )
        self.name = name
        self.rollout_length = rollout_length

        # Expert's buffer
        self.buffer_exp = buffer_exp

        # Cost function (simpler than AIRL discriminator)
        self.disc = RGCLCost(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units=units_disc,
            hidden_activation=nn.ReLU(inplace=False)
        ).to(device)

        # Get flattened parameter vector
        self.theta = self.disc.get_theta()
        n_theta = len(self.theta)
        self.n_features = n_theta

        # Kalman filter matrices
        self.P = P_init * torch.eye(n_theta, device=device)  # Covariance matrix
        self.Q = Q_init * torch.eye(n_theta, device=device)  # Process noise

        self.learning_steps_disc = 0
        self.batch_size = batch_size
        self.disc_update_freq = disc_update_freq

    def is_update(self, step, algo=None):
        """AVG updates every step"""
        return True

    def update(self, writer=None, step=None):
        """Update cost function and AVG policy"""
        self.learning_steps += 1

        # Update discriminator/cost function periodically
        if self.learning_steps % self.disc_update_freq == 0 and self.buffer._n >= 1:
            self.learning_steps_disc += 1

            # Get sample from current policy
            if step is not None and self.buffer._n >= step:
                idx = (step - 1) % self.buffer._n
                states, _, _, dones, next_states = \
                    self.buffer.get_sample(idx)
            else:
                states, _, _, dones, next_states = \
                    self.buffer.sample(1)

            # Get sample from expert
            if step is not None:
                idx_exp = (step - 1) % self.buffer_exp.buffer_size
                states_exp, actions_exp, _, dones_exp, next_states_exp = \
                    self.buffer_exp.get_sample(idx_exp)
            else:
                states_exp, actions_exp, _, dones_exp, next_states_exp = \
                    self.buffer_exp.sample(1)

            # Update cost function using Kalman filter
            self.update_disc(states, states_exp)

        # Get the most recent transition for AVG update
        if self.buffer._n >= self.batch_size:
            # Get recent batch for policy update
            end_idx = self.buffer._p
            available = min(self.buffer._n, self.batch_size)

            if available < self.batch_size:
                # Not enough samples yet
                if end_idx >= available:
                    idxes = slice(end_idx - available, end_idx)
                else:
                    start = self.buffer.buffer_size - (available - end_idx)
                    indices = list(range(start, self.buffer.buffer_size)) + \
                             list(range(0, end_idx))
                    idxes = indices
            else:
                start_idx = end_idx - self.batch_size
                if start_idx >= 0:
                    idxes = slice(start_idx, end_idx)
                else:
                    indices = list(range(start_idx + self.buffer.buffer_size,
                                       self.buffer.buffer_size)) + \
                             list(range(0, end_idx))
                    idxes = indices

            states, actions, _, dones, next_states = \
                self.buffer.get_sample(idxes)

            # Use first transition for online AVG update
            state = states[0] if states.dim() > 1 else states
            action = actions[0] if actions.dim() > 1 else actions
            done = dones[0].item() if dones.dim() > 0 else dones.item()
            log_pi = self.last_log_pi  # Use stored log_pi from step()
            next_state = next_states[0] if next_states.dim() > 1 else next_states

            # Calculate reward from RGCL cost function
            with torch.no_grad():
                # RGCL uses state-only cost
                reward = self.disc.calculate_reward(state.unsqueeze(0)).item()

            # Update AVG using cost-derived reward
            self.update_avg(state.unsqueeze(0), action.unsqueeze(0),
                          next_state.unsqueeze(0), reward, done, log_pi, writer)

    def update_disc(self, states, states_exp):
        """Update cost function parameters using Kalman filter"""
        self.theta = self.disc.get_theta()

        # Compute gradients and Hessians (Fisher approximation)
        grad_s, hessian_s = self._compute_grad_and_hessian(self.theta, states)
        grad_d, hessian_d = self._compute_grad_and_hessian(self.theta, states_exp)

        # Kalman update
        # Prediction: P_pred = P + Q
        P_pred = self.P + self.Q

        # Innovation: S = inv(P_pred) + H_d - H_s
        P_pred_inv = torch.linalg.inv(P_pred + 1e-6 * torch.eye(
            self.n_features, device=self.device))
        S = P_pred_inv + hessian_d - hessian_s

        # Update covariance: P_new = inv(S)
        P_new = torch.linalg.inv(S + 1e-6 * torch.eye(
            self.n_features, device=self.device))

        # Update parameters
        grad_diff = grad_d - grad_s
        update_step = torch.matmul(P_new, grad_diff)
        self.theta = self.theta - update_step

        self.P = P_new
        self.disc.set_theta(self.theta)

    def _compute_grad_and_hessian(self, theta_vec, state_input):
        """Compute gradient and Fisher Information Matrix approximation

        Fisher approximation: F ≈ g * g^T (outer product of gradients)
        This is much faster than computing the full Hessian.

        Args:
            theta_vec: Flattened parameter vector
            state_input: The state to evaluate cost on

        Returns:
            grads: (N_params,)
            fisher_approx: (N_params, N_params)
        """
        from torch.func import functional_call

        # Create parameter dict from theta_vec
        theta_input = theta_vec.detach().clone()
        theta_input.requires_grad = True

        # Convert flat theta to named parameter dict
        param_dict = {}
        offset = 0
        for name, param in self.disc.named_parameters():
            numel = param.numel()
            param_dict[name] = theta_input[offset:offset+numel].view(param.shape)
            offset += numel

        # Functional forward pass
        state_batch = state_input.unsqueeze(0) if state_input.dim() == 1 else state_input
        cost = functional_call(self.disc, param_dict, state_batch).squeeze()

        # Compute gradient
        grads = torch.autograd.grad(cost, theta_input)[0] / max(1, self.buffer._n)

        # Compute Fisher Information Matrix approximation
        # F ≈ g * g^T (outer product)
        fisher_approx = torch.outer(grads, grads)

        return grads.detach(), fisher_approx.detach()

    def save_models(self, save_dir):
        """Save actor, critic, and cost function"""
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'critic.pth'))
        torch.save(self.disc.state_dict(), os.path.join(save_dir, 'disc.pth'))
        torch.save({
            'theta': self.theta,
            'P': self.P,
            'Q': self.Q
        }, os.path.join(save_dir, 'kalman_state.pth'))
