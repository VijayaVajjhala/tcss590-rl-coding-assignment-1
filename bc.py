"""
TODO: MODIFY TO FILL IN YOUR BC IMPLEMENTATION
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50,
                       batch_size=32):

    # Fill in your BC implementation in this function.

    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy.
    flattened = {'observations': [], 'actions': []}
    for path in expert_data:
        for k in flattened.keys():
            flattened[k].append(path[k])
    for k in flattened.keys():
        flattened[k] = np.concatenate(flattened[k])

    obs_all = torch.from_numpy(flattened['observations']).float().to(device)
    acs_all = torch.from_numpy(flattened['actions']).float().to(device)
    N = obs_all.shape[0]

    optimizer = optim.Adam(list(policy.parameters()), lr=1e-4)
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs)*episode_length // batch_size
    losses = []
    flattened_idxs = np.concatenate([
                        np.arange(idx * episode_length, (idx + 1) * episode_length) 
                        for idx in idxs
                    ])
    for epoch in range(num_epochs):
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here, just maximize log likelihood!
            # Sample a minibatch of (obs, action) pairs, compute the negative log-likelihood
            # of the actions under the policy, and assign it to `loss`.

            batch_idxs = flattened_idxs[i * batch_size : (i + 1) * batch_size]

            obs_batch = obs_all[batch_idxs]
            acs_batch = acs_all[batch_idxs]

            log_probs = policy.log_prob(obs_batch, acs_batch)
            loss = -log_probs.mean()
  
            #loss = None
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 10 == 0:
        print('[%d] loss: %.8f' %
            (epoch, running_loss / 10.))
        #losses.append(loss.item())
        losses.append(running_loss/num_batches)
    return losses
