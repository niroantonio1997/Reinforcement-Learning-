import random
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, type='discrete'):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        if type == 'discrete':
            self.acts_buf = np.zeros(size, dtype=np.int64)
        else:
            self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.log_probs_buf = np.zeros(size, dtype=np.float32)  # for storing log probabilities if needed

    # store experience in the buffer (last in first out)
    def store(self, obs, act, rew, next_obs, done, log_prob):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        if log_prob is not None:
            self.log_probs_buf[self.ptr] = log_prob
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(states=self.obs_buf[idxs],
                    new_states=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs], 
                    log_probs=self.log_probs_buf[idxs] if hasattr(self, 'log_probs_buf') else None)
    
    def return_all(self):
        return dict(states=self.obs_buf[:self.size],
                    new_states=self.next_obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    rews=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size],
                    log_probs=self.log_probs_buf[:self.size] if self.log_probs_buf is not None else None)
    
    def clear(self):
        self.obs_buf = np.zeros((self.max_size, self.obs_buf.shape[1]), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.max_size, self.next_obs_buf.shape[1]), dtype=np.float32)
        if self.acts_buf.ndim == 1:
            self.acts_buf = np.zeros(self.max_size, dtype=self.acts_buf.dtype)
        else:
            self.acts_buf = np.zeros((self.max_size, self.acts_buf.shape[1]), dtype=np.float32)
        self.rews_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0
    