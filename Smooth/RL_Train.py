"""
RL_Train.py

This script trains a reinforcement learning agent using the BSGD, ABSGD, ABSGD-STORM algorithms
on the Pendulum-v1 environment from the OpenAI Gym.

Supports the paper 
Liu, Yin, and Sam Davanloo Tajbakhsh. "Adaptive Stochastic Optimization Algorithms for Problems with Biased Oracles." arXiv preprint arXiv:2306.07810 (2023).
https://arxiv.org/abs/2306.07810


Author: Yin Liu (liu.6630@osu.edu)
Date: 2023-10-05

"""


import copy
import numpy as np
import pickle
import torch
import torch.nn as nn
import gym
import scipy.signal
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import copy
from torch.distributions import Normal


env = gym.make("Pendulum-v1")
discount = 0.97


sample_batch = 5000

eta = np.linspace(2, 300, 20, dtype=int)
sample_variance_Relu = np.zeros(len(eta))
sample_mean_Relu = np.zeros([len(eta), 161])

for i in range(len(eta)):
    H = eta[i]  # maximum length
    grad_sample = np.zeros([sample_batch, 161])

    file_name = "Data/Relu/Pendulum_maximum300_gamma97_grad_H_" + str(int(H)) + ".pt"
    grad_sample = torch.load(file_name)
    print("finish eta = %d\r" % H, end="")
    sample_variance_Relu[i] = np.sum(np.var(grad_sample, axis=0))
    sample_mean_Relu[i, :] = np.mean(grad_sample, axis=0)
print("\n")
# fit hv
z = np.polyfit(eta[2:], sample_variance_Relu[2:], 6)
poly_var = np.poly1d(z)

# fit hb
sample_bias_Relu = (
    np.linalg.norm(sample_mean_Relu - sample_mean_Relu[-1, :], axis=1) ** 2
)
z_bias = np.polyfit(eta[:-1], np.log10(sample_bias_Relu[:-1]), 3)
poly_bias = np.poly1d(z_bias)


# %% define NN Policy
class GaussianPolicy(nn.Module):
    def __init__(self):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2

        return x


def get_policy(state, policy):
    # function that uses NN to calculate mean of normal distribution, returns the Normal objective

    state = torch.as_tensor(np.array(state), dtype=torch.float)
    mean = policy(state)
    return Normal(mean, 1)


def get_action(state, policy):
    # sample the Normal distribution and return the result as action

    action = get_policy(state, policy).sample()
    return action


def discount_cumsum(x: np.ndarray, gamma: float) -> torch.tensor:
    # copy from Ray v1.9.2 source code https://docs.ray.io/en/releases-1.9.2/_modules/ray/rllib/evaluation/postprocessing.html?highlight=discount_cumsum#
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma: The discount factor gamma.

    Returns:
        The sequence containing the discounted cumulative sums
        for each individual reward in `x` till the end of the trajectory.

    Examples:
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> gamma = 0.9
        >>> discount_cumsum(x, gamma)
        ... array([0.0 + 0.9*1.0 + 0.9^2*2.0 + 0.9^3*3.0,
        ...        1.0 + 0.9*2.0 + 0.9^2*3.0,
        ...        2.0 + 0.9*3.0,
        ...        3.0])
    """
    result = scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]
    return torch.as_tensor(result.copy())


def compute_objective(obs_traj, action_traj, reward_traj, policy, discount=0.97):
    H = len(reward_traj)
    Q_func = discount_cumsum(torch.Tensor.numpy(reward_traj), discount)
    weight = discount ** torch.arange(0, H)
    log_prob = get_policy(obs_traj, policy).log_prob(action_traj).reshape(-1)

    obj_func = Q_func * log_prob * weight
    return obj_func.sum()  # .mean()  #


def get_samples(policy, maximum_length=100):
    # get a single batch trajectory
    obs_traj = []
    action_traj = []
    reward_traj = []

    # reset the environment
    # obs, _ = env.reset(seed=6)
    obs, _ = env.reset(seed=0)

    for _ in range(maximum_length):
        # take one step
        action = get_action(obs, policy)  # stochastic action
        # state = torch.as_tensor(np.array(obs), dtype=torch.float)
        # action = policy(state).detach() # determinastic action
        next_state, reward, done, _, _ = env.step(action)

        # save the result
        obs_traj.append(obs.copy())
        action_traj.append(action)
        reward_traj.append(reward)

        # update obs
        obs = next_state

    eps = [obs_traj, action_traj, reward_traj]
    return eps


def get_grad(model):
    # get gradient of a NN model
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.reshape(-1))
    return torch.cat([g for g in temp_grad])


def calculate_grad(eps, policy, H):
    obs_traj = eps[0]
    action_traj = torch.as_tensor(eps[1]).reshape(-1, 1)
    reward_traj = torch.as_tensor(eps[2])
    loss = -compute_objective(
        obs_traj[:H], action_traj[:H], reward_traj[:H], policy, discount
    )
    loss.backward()


##############################################

# %% NB-SGD


def task_NBSGD(exp):
    alpha = 1e-3
    discount = 0.97
    H = 200
    H_max = 300
    batch_size = 100
    batch_size_inf = 200
    num_iter = 50
    torch.manual_seed(6)
    policy_init = GaussianPolicy()
    policy = copy.deepcopy(policy_init)

    torch.manual_seed(exp)

    H_iteration = np.zeros(num_iter)
    dis_reward_iteration = np.zeros(num_iter)
    grad_norm_iteration = np.zeros(num_iter)

    for ite in range(num_iter):
        policy.zero_grad()
        for _ in range(batch_size):
            eps = get_samples(policy, maximum_length=H)
            calculate_grad(eps, policy, H)

        # divide gradient by batch_size
        with torch.no_grad():
            grad = get_grad(policy)
            grad = grad / batch_size

        # calculate grad_inf
        discount_reward = 0
        policy.zero_grad()
        for _ in range(batch_size_inf):
            eps = get_samples(policy, maximum_length=H_max)
            calculate_grad(eps, policy, H=H_max)
            temp = discount ** torch.arange(0, H_max) * torch.as_tensor(eps[2])
            discount_reward = temp.sum() + discount_reward

        with torch.no_grad():
            grad_inf = get_grad(policy)
            grad_inf = grad_inf / batch_size_inf

        H_iteration[ite] = H
        dis_reward_iteration[ite] = discount_reward / batch_size_inf
        grad_norm_iteration[ite] = np.linalg.norm(grad_inf) ** 2
        print(ite, dis_reward_iteration[ite], grad_norm_iteration[ite], end="\n")
        param = parameters_to_vector(policy.parameters())
        param = param - alpha * grad
        vector_to_parameters(param, policy.parameters())

        # print('exp=', exp, 'ite =', ite, 'reward =',
        #      dis_reward_iteration[ite], 'norm_grad=', grad_norm_iteration[ite], end='\r')
    print(exp)
    return (
        exp,
        dis_reward_iteration,
        grad_norm_iteration,
        H_iteration,
    )


# %% adaptive algorithm


def task_ABSGD(exp):
    alpha = 1e-3
    discount = 0.97
    H = 50
    batch_size = 100
    num_iter = 50
    H_max = 300
    batch_size_inf = 200
    torch.manual_seed(6)
    policy_init = GaussianPolicy()
    policy = copy.deepcopy(policy_init)
    torch.manual_seed(exp)

    H_iteration = np.zeros(num_iter)
    dis_reward_iteration = np.zeros(num_iter)
    grad_norm_iteration = np.zeros(num_iter)

    for ite in range(num_iter):
        # determine the bias
        while True:
            policy.zero_grad()
            for _ in range(batch_size):
                eps = get_samples(policy, maximum_length=H)
                calculate_grad(eps, policy, H)

            with torch.no_grad():
                grad = get_grad(policy)
                grad = grad / batch_size
            if H == 200 or 10 ** poly_bias(H) <= 0.008 * np.linalg.norm(grad) ** 2:
                break
            else:
                H = H + 10
                if H > 200:
                    H = 200
                    break

        H_iteration[ite] = H

        # calculate grad_inf
        discount_reward = 0
        policy.zero_grad()
        for _ in range(batch_size_inf):
            eps = get_samples(policy, maximum_length=H_max)
            calculate_grad(eps, policy, H=H_max)
            temp = discount ** torch.arange(0, H_max) * torch.as_tensor(eps[2])
            discount_reward = temp.sum() + discount_reward

        with torch.no_grad():
            grad_inf = get_grad(policy)
            grad_inf = grad_inf / batch_size_inf

        dis_reward_iteration[ite] = discount_reward / batch_size_inf
        grad_norm_iteration[ite] = np.linalg.norm(grad_inf) ** 2

        param = parameters_to_vector(policy.parameters())
        param = param - alpha * grad
        vector_to_parameters(param, policy.parameters())

        print(
            ite,
            dis_reward_iteration[ite],
            grad_norm_iteration[ite],
            H_iteration[ite],
            end="\n",
        )
    print(exp)

    return (exp, dis_reward_iteration, grad_norm_iteration, H_iteration)
    # print('exp=', exp, 'ite =', ite, 'reward =',
    # dis_reward_iteration[ite], 'norm_grad=', grad_norm_iteration[ite], end='\r')


# %% AB-STORM fixed step


def task_ABST(exp):
    alpha = 1e-5
    beta = 0.5
    discount = 0.97
    H = 50
    batch_size = 50
    num_iter = 50
    H_max = 300
    batch_size_inf = 200

    torch.manual_seed(6)
    policy_init = GaussianPolicy()
    policy = copy.deepcopy(policy_init)
    torch.manual_seed(exp)

    H_iteration = np.zeros(num_iter)

    dis_reward_iteration = np.zeros(num_iter)
    grad_norm_iteration = np.zeros(num_iter)

    # k = 1
    while True:
        policy.zero_grad()
        for _ in range(batch_size):
            eps = get_samples(policy, maximum_length=H)
            calculate_grad(eps, policy, H)

        with torch.no_grad():
            grad = get_grad(policy)
            g = grad / batch_size

        if H == 200 or 10 ** poly_bias(H) <= 0.008 * np.linalg.norm(g) ** 2:
            break
        else:
            H = H + 10
            if H > 200:
                H = 200
                break

    for ite in range(num_iter):
        H_iteration[ite] = H
        # calculate grad_inf
        discount_reward = 0
        policy.zero_grad()
        for _ in range(batch_size_inf):
            eps = get_samples(policy, maximum_length=300)
            calculate_grad(eps, policy, 300)
            temp = discount ** torch.arange(0, 300) * torch.as_tensor(eps[2])
            discount_reward = temp.sum() + discount_reward

        with torch.no_grad():
            grad_inf = copy.deepcopy(get_grad(policy))
            grad_inf = grad_inf / batch_size_inf

        dis_reward_iteration[ite] = discount_reward / batch_size_inf
        grad_norm_iteration[ite] = np.linalg.norm(grad_inf) ** 2

        alphak = alpha
        betak = beta
        policy_pre = copy.deepcopy(policy)
        H_pre = copy.deepcopy(H)

        param = parameters_to_vector(policy.parameters())
        param = param - alphak * g
        vector_to_parameters(param, policy.parameters())

        while True:
            if H == 200 or 10 ** poly_bias(H) <= 0.008 * np.linalg.norm(g) ** 2:
                break
            else:
                H = H + 10
                if H > 200:
                    H = 200
                    break

        # get gk
        policy.zero_grad()
        policy_pre.zero_grad()

        for _ in range(batch_size):
            eps = get_samples(policy, maximum_length=H)
            calculate_grad(eps, policy, H)

            for i in range(len(eps)):
                eps[i] = eps[i][0:H_pre]
            calculate_grad(eps, policy_pre, H_pre)

        with torch.no_grad():
            grad = copy.deepcopy(get_grad(policy))
            grad_pre = copy.deepcopy(get_grad(policy_pre))
            grad = grad / batch_size
            grad_pre = grad_pre / batch_size

            g = grad + (1 - betak) * (g - grad_pre)
        print(
            ite,
            dis_reward_iteration[ite],
            grad_norm_iteration[ite],
            H_iteration[ite],
            end="\n",
        )
    print(exp)
    return (exp, dis_reward_iteration, grad_norm_iteration, H_iteration)


# %% AB-STORM varying step
def task_ABSTD(exp):
    alpha = 1e-4
    beta = 0.5
    discount = 0.97
    H = 50
    batch_size = 50
    num_iter = 50
    H_max = 300
    batch_size_inf = 200

    torch.manual_seed(6)
    policy_init = GaussianPolicy()
    policy = copy.deepcopy(policy_init)
    torch.manual_seed(exp)

    H_iteration = np.zeros(num_iter)

    dis_reward_iteration = np.zeros(num_iter)
    grad_norm_iteration = np.zeros(num_iter)

    # k = 1
    while True:
        policy.zero_grad()
        for _ in range(batch_size):
            eps = get_samples(policy, maximum_length=H)
            calculate_grad(eps, policy, H)

        with torch.no_grad():
            grad = get_grad(policy)
            g = grad / batch_size

        if H == 200 or 10 ** poly_bias(H) <= 0.008 * np.linalg.norm(g) ** 2:
            break
        else:
            H = H + 10
            if H > 200:
                H = 200
                break

    for ite in range(num_iter):
        H_iteration[ite] = H
        # calculate grad_inf
        discount_reward = 0
        policy.zero_grad()
        for _ in range(batch_size_inf):
            eps = get_samples(policy, maximum_length=300)
            calculate_grad(eps, policy, 300)
            temp = discount ** torch.arange(0, 300) * torch.as_tensor(eps[2])
            discount_reward = temp.sum() + discount_reward

        with torch.no_grad():
            grad_inf = copy.deepcopy(get_grad(policy))
            grad_inf = grad_inf / batch_size_inf

        dis_reward_iteration[ite] = discount_reward / batch_size_inf
        grad_norm_iteration[ite] = np.linalg.norm(grad_inf) ** 2

        alphak = alpha / (50 * ite + 1) ** (1 / 3)
        betak = beta
        policy_pre = copy.deepcopy(policy)
        H_pre = copy.deepcopy(H)

        param = parameters_to_vector(policy.parameters())
        param = param - alphak * g
        vector_to_parameters(param, policy.parameters())

        while True:
            if H == 200 or 10 ** poly_bias(H) <= 0.008 * np.linalg.norm(g) ** 2:
                break
            else:
                H = H + 10
                if H > 200:
                    H = 200
                    break

        # get gk
        policy.zero_grad()
        policy_pre.zero_grad()

        for _ in range(batch_size):
            eps = get_samples(policy, maximum_length=H)
            calculate_grad(eps, policy, H)

            for i in range(len(eps)):
                eps[i] = eps[i][0:H_pre]
            calculate_grad(eps, policy_pre, H_pre)

        with torch.no_grad():
            grad = copy.deepcopy(get_grad(policy))
            grad_pre = copy.deepcopy(get_grad(policy_pre))
            grad = grad / batch_size
            grad_pre = grad_pre / batch_size

            g = grad + (1 - betak) * (g - grad_pre)
        print(
            ite,
            dis_reward_iteration[ite],
            grad_norm_iteration[ite],
            H_iteration[ite],
            end="\n",
        )
    print(exp)
    return (exp, dis_reward_iteration, grad_norm_iteration, H_iteration)


# %%
# entry point for the program
if __name__ == "__main__":
    num_exp = 20
    num_iter = 50

    dis_reward_NBSGD = np.zeros((num_exp, num_iter))
    grad_norm_NBSGD = np.zeros((num_exp, num_iter))
    batch_size_NBSGD = np.zeros((num_exp, num_iter))

    for exp in range(num_exp):
        result = task_NBSGD(exp)
        dis_reward_NBSGD[result[0], :] = result[1]
        grad_norm_NBSGD[result[0], :] = result[2]
        batch_size_NBSGD[result[0], :] = result[3]

        with open(
            "Data/RL_parallel_NBSGD_20ite.pkl", "wb"
        ) as f:  # Python 3: open(..., 'wb')
            pickle.dump([dis_reward_NBSGD, grad_norm_NBSGD, batch_size_NBSGD], f)

    dis_reward_ABSGD = np.zeros((num_exp, num_iter))
    grad_norm_ABSGD = np.zeros((num_exp, num_iter))
    H_ABSGD = np.zeros((num_exp, num_iter))
    for exp in range(num_exp):
        result = task_ABSGD(exp)
        dis_reward_ABSGD[result[0], :] = result[1]
        grad_norm_ABSGD[result[0], :] = result[2]
        H_ABSGD[result[0], :] = result[3]

        with open("Data/RL_parallel_ABSGD_20ite.pkl", "wb") as f:
            pickle.dump([dis_reward_ABSGD, grad_norm_ABSGD, H_ABSGD], f)

    dis_reward_ABST = np.zeros((num_exp, num_iter))
    grad_norm_ABST = np.zeros((num_exp, num_iter))
    H_ABST = np.zeros((num_exp, num_iter))
    for exp in range(num_exp):
        result = task_ABST(exp)
        dis_reward_ABST[result[0], :] = result[1]
        grad_norm_ABST[result[0], :] = result[2]
        H_ABST[result[0], :] = result[3]

        with open("Data/RL_parallel_ABST_20ite.pkl", "wb") as f:
            pickle.dump([dis_reward_ABST, grad_norm_ABST, H_ABST], f)

    dis_reward_ABSTD = np.zeros((num_exp, num_iter))
    grad_norm_ABSTD = np.zeros((num_exp, num_iter))
    H_ABSTD = np.zeros((num_exp, num_iter))
    for exp in range(num_exp):
        result = task_ABSTD(exp)
        dis_reward_ABSTD[result[0], :] = result[1]
        grad_norm_ABSTD[result[0], :] = result[2]
        H_ABSTD[result[0], :] = result[3]

        with open("Data/RL_parallel_ABSTD_20ite.pkl", "wb") as f:
            pickle.dump([dis_reward_ABSTD, grad_norm_ABSTD, H_ABSTD], f)
