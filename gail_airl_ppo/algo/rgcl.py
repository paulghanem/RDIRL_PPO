import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam,SGD

from .ppo import PPO
from gail_airl_ppo.network import AIRLDiscrim,RGCLCost
import pdb


class RGCL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=2, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-5,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(64, 64), units_disc_v=(100, 100),
                 epoch_ppo=1, epoch_disc=1, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0,name=None):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )
        self.name=name

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = RGCLCost(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units=units_disc,
            hidden_activation=nn.ReLU(inplace=False)
        ).to(device)
        
        self.theta=self.disc.get_theta()


        #pdb.set_trace()
        n_theta = len(self.theta)
        self.n_features = n_theta  # Store for later use

        self.P =1e-4* torch.eye(n_theta)

        self.Q =1e-6*torch.eye(n_theta)
  


       

        self.learning_steps_disc = 0
        self.batch_size = batch_size
        

    def update(self, step):

        self.learning_steps += 1

        # Skip PPO update if we don't have enough samples yet
        if self.buffer._n < self.batch_size:
            # Still update discriminator
            self.learning_steps_disc += 1
            states, _, _, dones, log_pis, next_states = \
                self.buffer.get_sample((step-1)%self.rollout_length)
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.get_sample((step-1) % self.buffer_exp.buffer_size)
            self.update_disc(states, states_exp)
            return  # Skip PPO update

        # if step%self.rollout_length==0:
        #      self.P=1e-1
            
        self.learning_steps_disc += 1

        # Samples from current policy's trajectories.
        # states, _, _, dones, log_pis, next_states = \
        #     self.buffer.sample(self.batch_size)
        states, _, _, dones, log_pis, next_states = \
            self.buffer.get_sample((step-1)%self.rollout_length)
        # Samples from expert's demonstrations.
        # states_exp, actions_exp, _, dones_exp, next_states_exp = \
        #     self.buffer_exp.sample(self.batch_size)
        states_exp, actions_exp, _, dones_exp, next_states_exp = \
            self.buffer_exp.get_sample((step-1) % self.buffer_exp.buffer_size)
        # Calculate log probabilities of expert actions.
        # Update discriminator.
        self.update_disc(
            states, states_exp
        )

        # We don't use reward signals here,
        # Use sample instead of get for RGCL per-step updates
        #if step % self.rollout_length == 0:

        # Get last batch_size samples (handles circular buffer wraparound)
        end_idx = self.buffer._p
        available = min(self.buffer._n, self.batch_size)  # Don't try to get more than we have

       
        
        if available < self.batch_size:
            # Not enough samples yet - get what we have
            if end_idx >= available:
                idxes = slice(end_idx - available, end_idx)
            else:
                # Wraparound even with partial buffer
                start = self.buffer.total_size - (available - end_idx)
                indices = list(range(start, self.buffer.total_size)) + list(range(0, end_idx))
                idxes = indices
        else:
            # Have enough samples - get last batch_size
            start_idx = end_idx - self.batch_size
            if start_idx >= 0:
                idxes = slice(start_idx, end_idx)
            else:
                indices = list(range(start_idx + self.buffer.total_size, self.buffer.total_size)) + \
                          list(range(0, end_idx))
                idxes = indices
                
                
        #if step%self.rollout_length==0:
        states, actions, _, dones, log_pis, next_states = self.buffer.get_sample(idxes)
        #states, actions, _, dones, log_pis, next_states = self.buffer.get()
        # Calculate rewards (RGCL only needs states).
        rewards = self.disc.calculate_reward(states)
    
        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer=None)
        

    def update_disc(self, states,
                    states_exp):
        
       
        
        self.theta=self.disc.get_theta()
        # Instead of separate calls, we get full H and g
        # --- For Sampled State ---
        grad_s, hessian_s = self._compute_grad_and_hessian(self.theta, states)
        
        # --- For Expert State ---
        grad_d, hessian_d = self._compute_grad_and_hessian(self.theta, states_exp)
        
        # --- Kalman Update (Full Matrix) ---
       
        # Full Matrix Math
        # P_pred = P + Q
        P_pred = self.P + self.Q
        
        # Innovation S = inv(P_pred) + H_d - H_s
        # Note: The Hessian difference (Hd - Hs) measures curvature difference.
        # In IRL, positive curvature at expert (Hd) means expert is at a minimum (good).
        
        P_pred_inv = torch.linalg.inv(P_pred)
        S = P_pred_inv + hessian_d - hessian_s
        
        # Regularize and Invert
      
        P_new = torch.linalg.inv(S)
        
        # Update Mean
        grad_diff = grad_d - grad_s
        update_step = torch.matmul(P_new, grad_diff)
        self.theta = self.theta -  update_step
        self.P = P_new
        self.disc.set_theta(self.theta)
    
    def _compute_grad_and_hessian(self, theta_vec, state_input):
            """
            Computes Gradient and Fisher Information Matrix approximation.

            Fisher approximation: F ≈ g * g^T (outer product of gradients)
            This is much faster than computing the full Hessian.

            Args:
                theta_vec: Flattened parameter vector
                state_input: The state to evaluate cost on

            Returns:
                grads: (N_params,)
                fisher_approx: (N_params, N_params) - Fisher Information Matrix approximation
            """
            from torch.func import functional_call

            # 1. Create parameter dict from theta_vec
            theta_input = theta_vec.detach().clone()
            theta_input.requires_grad = True

            # Convert flat theta to named parameter dict
            param_dict = {}
            offset = 0
            for name, param in self.disc.named_parameters():
                numel = param.numel()
                param_dict[name] = theta_input[offset:offset+numel].view(param.shape)
                offset += numel

            # 2. Functional forward pass
            state_batch = state_input.unsqueeze(0)
            cost = functional_call(self.disc, param_dict, state_batch).squeeze()

            # 3. Compute Gradient
            grads = torch.autograd.grad(cost, theta_input)[0]/self.rollout_length

            # 4. Compute Fisher Information Matrix approximation
            # F ≈ g * g^T (outer product)
            # This is a rank-1 approximation, much faster than full Hessian
            fisher_approx = torch.outer(grads, grads)

            return grads.detach(), fisher_approx.detach()

