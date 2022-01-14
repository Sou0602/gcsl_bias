import numpy as np
import gym
import pdb
import rlutil.torch as torch
import rlutil.torch.distributions
import rlutil.torch.nn as nn
import torch.nn.functional as F
import rlutil.torch.pytorch_util as ptu
from torch.nn.parameter import Parameter

from gcsl import policy


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FCNetwork(nn.Module):
    """
    A fully-connected network module
    """

    def __init__(self, dim_input, dim_output, layers=[256, 256],
                 nonlinearity=torch.nn.ReLU, dropout=0):
        super(FCNetwork, self).__init__()
        net_layers = []
        dim = dim_input
        for i, layer_size in enumerate(layers):
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(nonlinearity())
            if dropout > 0:
                net_layers.append(torch.nn.Dropout(0.4))
            dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers
        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, states):
        return self.network(states)

class FCNetwork_m(nn.Module):
    """
    A fully-connected network module
    """

    def __init__(self, dim_input, dim_output, layers=[256, 256],
                 nonlinearity=torch.nn.ReLU, dropout=0):
        super(FCNetwork_m, self).__init__()
        net_layers = []
        dim = dim_input
        for i, layer_size in enumerate(layers):
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(nonlinearity())
            if dropout > 0:
                net_layers.append(torch.nn.Dropout(0.4))
            dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers
        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, states):
        return self.network(states)

class FCNetwork_n(nn.Module):
    """
    A fully-connected network module
    """

    def __init__(self, dim_input, dim_output, layers=[256, 256],
                 nonlinearity=torch.nn.ReLU, dropout=0):
        super(FCNetwork_n, self).__init__()
        net_layers = torch.nn.ModuleList([])
        dim = dim_input
        self.dim_output = dim_output
        for i, layer_size in enumerate(layers):
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(nonlinearity())
            if dropout > 0:
                net_layers.append(torch.nn.Dropout(0.4))
            dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.net_layers = net_layers
        #self.layers = net_layers
        #self.network1 = torch.nn.Sequential(*net_layers)

        dim_1 = int(dim_input / 2)
        net_layers_1 = torch.nn.ModuleList([])
        for i, layer_size in enumerate(layers):
            net_layers_1.append(torch.nn.Linear(dim_1, layer_size))
            net_layers_1.append(nonlinearity())
            if dropout > 0:
                net_layers_1.append(torch.nn.Dropout(0.4))
            dim_1 = layer_size
        net_layers_1.append(torch.nn.Linear(dim, dim_output))
        self.net_layers_1 = net_layers_1

    def forward(self, states):
        #print(states)
        out_sg = states
        for i in range(len(self.net_layers)):
            out_sg = self.net_layers[i](out_sg)
        #print(out_sg)


        out_s = states[:,:int(states.shape[1]/2)]
        #out_s = torch.reshape(out_s,(out_sg.shape[0],-1))

        for i in range(len(self.net_layers_1)):
            out_s = self.net_layers_1[i](out_s)
       # pdb.set_trace()
        #print(out_s)
        #print(out_sg.shape , out_s.shape)
        return torch.cat((out_sg,out_s),dim=1)



class CBCNetwork(nn.Module):
    """
    A fully connected network which appends conditioning to each hidden layer
    """

    def __init__(self, dim_input, dim_conditioning, dim_output, layers=[256, 256],
                 nonlinearity=torch.nn.ReLU, dropout=0, add_conditioning=True):
        super(CBCNetwork, self).__init__()

        self.dropout = bool(dropout != 0)
        self.add_conditioning = add_conditioning

        net_layers = torch.nn.ModuleList([])
        dim = dim_input + dim_conditioning

        for i, layer_size in enumerate(layers):
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(nonlinearity())
            if self.dropout:
                net_layers.append(torch.nn.Dropout(dropout))
            if add_conditioning:
                dim = layer_size + dim_conditioning
            else:
                dim = layer_size

        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers

    def forward(self, states, conditioning):
        output = torch.cat((states, conditioning), dim=1)
        mod = 3 if self.dropout else 2
        for i in range(len(self.layers)):
            output = self.layers[i](output)
            if i % mod == mod - 1 and self.add_conditioning:
                output = torch.cat((output, conditioning), dim=1)
        return output


class MultiInputNetwork(nn.Module):
    def __init__(self, input_shapes, dim_out, input_embeddings=None, layers=[512, 512], freeze_embeddings=False):
        super(MultiInputNetwork, self).__init__()
        if input_embeddings is None:
            input_embeddings = [Flatten() for _ in range(len(input_shapes))]

        self.input_embeddings = input_embeddings
        self.freeze_embeddings = freeze_embeddings

        dim_ins = [
            embedding(torch.tensor(np.zeros((1,) + input_shape))).size(1)
            for embedding, input_shape in zip(input_embeddings, input_shapes)
        ]

        full_dim_in = sum(dim_ins)
        self.net = FCNetwork(full_dim_in, dim_out, layers=layers)

    def forward(self, *args):
        assert len(args) == len(self.input_embeddings)
        embeddings = [embed_fn(x) for embed_fn, x in zip(self.input_embeddings, args)]
        embed = torch.cat(embeddings, dim=1)
        if self.freeze_embeddings:
            embed = embed.detach()
        return self.net(embed)


class StateGoalNetwork(nn.Module):
    def __init__(self, env, dim_out=1, state_embedding=None, goal_embedding=None, layers=[512, 512], max_horizon=None,
                 freeze_embeddings=False, add_extra_conditioning=False, dropout=0):
        super(StateGoalNetwork, self).__init__()
        self.max_horizon = max_horizon
        if state_embedding is None:
            state_embedding = Flatten()
        if goal_embedding is None:
            goal_embedding = Flatten()

        self.state_embedding = state_embedding
        self.goal_embedding = goal_embedding
        self.freeze_embeddings = freeze_embeddings

        state_dim_in = self.state_embedding(torch.tensor(torch.zeros(env.observation_space.shape)[None])).size()[1]
        goal_dim_in = self.goal_embedding(torch.tensor(torch.zeros(env.goal_space.shape)[None])).size()[1]

        dim_in = state_dim_in + goal_dim_in

        if max_horizon is not None:
            self.net = CBCNetwork(dim_in, max_horizon, dim_out, layers=layers, add_conditioning=add_extra_conditioning,
                                  dropout=dropout)
        else:
            self.net = FCNetwork(dim_in, dim_out, layers=layers)

    def forward(self, state, goal, horizon=None):
        state = self.state_embedding(state)
        goal = self.goal_embedding(goal)
        embed = torch.cat((state, goal), dim=1)
        if self.freeze_embeddings:
            embed = embed.detach()

        if self.max_horizon is not None:
            horizon = self.process_horizon(horizon)
            output = self.net(embed, horizon)
        else:
            output = self.net(embed)
        return output

    def process_horizon(self, horizon):
        # Todo add format options
        return horizon

class StateGoalNetwork_m(nn.Module):
    def __init__(self, env, dim_out=1, state_embedding=None, goal_embedding=None, layers=[512, 512], max_horizon=None,
                 freeze_embeddings=False, add_extra_conditioning=False, dropout=0):
        super(StateGoalNetwork_m, self).__init__()
        self.max_horizon = max_horizon
        if state_embedding is None:
            state_embedding = Flatten()
        if goal_embedding is None:
            goal_embedding = Flatten()

        self.state_embedding = state_embedding
        self.goal_embedding = goal_embedding
        self.freeze_embeddings = freeze_embeddings

        state_dim_in = self.state_embedding(torch.tensor(torch.zeros(env.observation_space.shape)[None])).size()[1]
        goal_dim_in = self.goal_embedding(torch.tensor(torch.zeros(env.goal_space.shape)[None])).size()[1]

        dim_in = state_dim_in

        if max_horizon is not None:
            pdb.set_trace()
            self.net = CBCNetwork(dim_in, max_horizon, dim_out, layers=layers, add_conditioning=add_extra_conditioning,
                                  dropout=dropout)
        else:
            self.net = FCNetwork_m(dim_in, dim_out, layers=layers)

    def forward(self, state, goal, horizon=None):
        state = self.state_embedding(state)
        goal = self.goal_embedding(goal)
        #embed = torch.cat((state, goal), dim=1)
        embed = state
        if self.freeze_embeddings:
            embed = embed.detach()

        if self.max_horizon is not None:
            horizon = self.process_horizon(horizon)
            output = self.net(embed, horizon)
        else:
            output = self.net(embed)
        return output

    def process_horizon(self, horizon):
        # Todo add format options
        return horizon




class StateGoalNetwork_n(nn.Module):
    def __init__(self, env, dim_out=1, state_embedding=None, goal_embedding=None, layers=[512, 512], max_horizon=None,
                 freeze_embeddings=False, add_extra_conditioning=False, dropout=0):
        super(StateGoalNetwork_n, self).__init__()
        self.max_horizon = max_horizon
        if state_embedding is None:
            state_embedding = Flatten()
        if goal_embedding is None:
            goal_embedding = Flatten()

        self.state_embedding = state_embedding
        self.goal_embedding = goal_embedding
        self.freeze_embeddings = freeze_embeddings

        state_dim_in = self.state_embedding(torch.tensor(torch.zeros(env.observation_space.shape)[None])).size()[1]
        goal_dim_in = self.goal_embedding(torch.tensor(torch.zeros(env.goal_space.shape)[None])).size()[1]

        dim_in = state_dim_in + goal_dim_in

        if max_horizon is not None:
            self.net = CBCNetwork(dim_in, max_horizon, dim_out, layers=layers, add_conditioning=add_extra_conditioning,
                                  dropout=dropout)
        else:
            self.net = FCNetwork_n(dim_in, dim_out, layers=layers)

    def forward(self, state, goal, horizon=None):
        state = self.state_embedding(state)
        goal = self.goal_embedding(goal)
        embed = torch.cat((state, goal), dim=1)
        if self.freeze_embeddings:
            embed = embed.detach()

        if self.max_horizon is not None:
            horizon = self.process_horizon(horizon)
            output = self.net(embed, horizon)
        else:
            output = self.net(embed)
        return output

    def process_horizon(self, horizon):
        # Todo add format options
        return horizon

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    one_hot_mask = (torch.arange(0, num_classes)
                    .long()
                    .repeat(batch_size, 1)
                    .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None, label_smoothing=0):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = torch.logsumexp(logits, dim=1) - (1 - label_smoothing) * class_select(logits,
                                                                                 target) - label_smoothing * logits.mean(
        dim=1)
    if loss.isnan().any():
        pdb.set_trace()
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """

    def __init__(self, aggregate='mean', label_smoothing=0):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.label_smoothing = label_smoothing

    def forward(self, input, target, weights=None):
        ce = cross_entropy_with_weights(input, target, weights, self.label_smoothing)
        if self.aggregate == 'sum':
            return ce.sum()
        elif self.aggregate == 'mean':
            return ce.mean()
        elif self.aggregate is None:
            return ce


class DiscreteStochasticGoalPolicy_m(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, **kwargs):
        super(DiscreteStochasticGoalPolicy_m, self).__init__()

        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = StateGoalNetwork(env, dim_out=self.dim_out, **kwargs)
        self.marg_net = StateGoalNetwork_m(env, dim_out=self.dim_out , **kwargs)
        self.t_marg_net = StateGoalNetwork_m(env, dim_out=self.dim_out , **kwargs)
        self.cond_loss = 0
        self.marg_loss = 0
    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def marg_forward(self,obs,goal,horizon = None):
        return self.marg_net.forward(obs,goal, horizon = horizon)

    def t_marg_forward(self,obs,goal,horizon = None):
        return self.t_marg_net.forward(obs,goal, horizon = horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
                       marginal_policy=None):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)

        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)

        logits = self.forward(obs, goal, horizon=horizon)
        marginal_logits = self.t_marg_forward(obs, goal, horizon=horizon)
        #logits -= marginal_logits
        eps = 0.00001
        eps1 = 0.00002
        # logits -= marginal_logits
        noisy_logits = logits * (1 - noise)
        m_noisy_logits = marginal_logits * (1)
        probs_sg = torch.softmax(noisy_logits, 1) + eps
        probs_s = torch.softmax(m_noisy_logits, 1) + eps1
        probs = torch.div(probs_sg, probs_s)
        #pdb.set_trace()
        probs = probs/torch.sum(probs,dim = 1)
        #logits = torch.logit(probs,eps = 1e-6)
        #noisy_logits = logits
        #probs = torch.softmax(noisy_logits, 1)

        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        return ptu.to_numpy(samples)

    def nll(self, obs, goal, actions, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        marginal_logits = self.marg_forward(obs, goal, horizon=horizon)
        #pdb.set_trace()
       # prob_logits = torch.softmax(logits,1)
       # prob_marginal_logits = torch.softmax(marginal_logits,1)
        #avg_prob = (prob_logits)*0.5 + (prob_marginal_logits)*0.5
        #loss_logits = torch.logit(avg_prob,eps=1e-6)
        self.cond_loss = CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits, actions, weights=None, )
        self.marg_loss = CrossEntropyLoss(aggregate=None, label_smoothing=0)(marginal_logits,actions,weights=None, )

        return CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits, actions, weights=None, ) + \
               CrossEntropyLoss(aggregate=None, label_smoothing=0)(marginal_logits,actions,weights=None, )

    def probabilities(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        return probs

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        Z = torch.logsumexp(logits, dim=1)
        return Z - torch.sum(probs * logits, 1)

    def process_horizon(self, horizon):
        return horizon

    def soft_update(self,tau):
        local_model = self.marg_net.net
        target_model = self.t_marg_net.net

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class DiscreteStochasticGoalPolicy_n(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, **kwargs):
        super(DiscreteStochasticGoalPolicy_n, self).__init__()

        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = StateGoalNetwork_n(env, dim_out=self.dim_out, **kwargs)
        #self.marg_net = StateMargNetwork(env, dim_out=self.dim_out, **kwargs)
    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
                       marginal_policy=None):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)

        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)

        logits = self.forward(obs, goal, horizon=horizon)[:,:self.dim_out]
        marginal_logits = self.forward(obs,goal,horizon=horizon)[:,self.dim_out:]
        eps = 0.00001
        eps1 = 0.00002
        #logits -= marginal_logits
        noisy_logits = logits * (1 - noise)
        m_noisy_logits = marginal_logits * (1)
        probs_sg = torch.softmax(noisy_logits, 1) + eps
        probs_s = torch.softmax(m_noisy_logits,1) + eps1
        probs = torch.div(probs_sg,probs_s)/torch.sum(probs,dim = 1)
        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()

        return ptu.to_numpy(samples)

    def nll(self, obs, goal, actions, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)[:,:self.dim_out]
        marginal_logits = self.forward(obs, goal, horizon=horizon)[:,self.dim_out:]
        #pdb.set_trace()
        prob_logits = torch.softmax(logits,1)
        prob_marginal_logits = torch.softmax(marginal_logits,1)
        avg_prob = (prob_logits + prob_marginal_logits)/2
        loss_logits = torch.logit(avg_prob,eps=1e-6)
        if loss_logits.isinf().any():
            pdb.set_trace()
        return CrossEntropyLoss(aggregate=None, label_smoothing=0)(loss_logits, actions, weights=None, )

    def probabilities(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)[:,:self.dim_out]
        probs = torch.softmax(logits, 1)
        return probs

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)[:,:self.dim_out]
        probs = torch.softmax(logits, 1)
        Z = torch.logsumexp(logits, dim=1)
        return Z - torch.sum(probs * logits, 1)

    def process_horizon(self, horizon):
        return horizon


class IndependentDiscretizedStochasticGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, **kwargs):
        super(IndependentDiscretizedStochasticGoalPolicy, self).__init__()

        self.action_space = env.action_space
        self.n_dims = self.action_space.n_dims
        self.granularity = self.action_space.granularity
        dim_out = self.n_dims * self.granularity
        self.net = StateGoalNetwork(env, dim_out=dim_out, **kwargs)

    def flattened(self, tensor):
        # tensor expected to be n x self.n_dims
        multipliers = self.granularity ** torch.tensor(np.arange(self.n_dims))
        flattened = (tensor * multipliers).sum(1)
        return flattened.int()

    def unflattened(self, tensor):
        # tensor expected to be n x 1
        digits = []
        output = tensor
        for _ in range(self.n_dims):
            digits.append(output % self.granularity)
            output = output // self.granularity
        uf = torch.stack(digits, dim=-1)
        return uf

    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0, marginal_policy=None):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)

        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)

        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        noisy_logits = logits * (1 - noise)
        probs = torch.softmax(noisy_logits, 2)

        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        samples = self.flattened(samples)
        if greedy:
            samples = ptu.to_numpy(samples)
            random_samples = np.random.choice(self.action_space.n, size=len(samples))
            return np.where(np.random.rand(len(samples)) < noise,
                            random_samples,
                            samples,
                            )
        return ptu.to_numpy(samples)

    def nll(self, obs, goal, actions, horizon=None):
        actions_perdim = self.unflattened(actions)
        # print(actions, self.flattened(actions_perdim))
        actions_perdim = actions_perdim.view(-1)

        logits = self.forward(obs, goal, horizon=horizon)
        logits_perdim = logits.view(-1, self.granularity)

        loss_perdim = CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits_perdim, actions_perdim, weights=None)
        loss = loss_perdim.reshape(-1, self.n_dims)
        return loss.sum(1)

    def probabilities(self, obs, goal, horizon=None):
        """
        TODO(dibyaghosh): actually implement
        """
        raise NotImplementedError()

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        probs = torch.softmax(noisy_logits, 2)
        Z = torch.logsumexp(logits, dim=2)
        return (Z - torch.sum(probs * logits, 2)).sum(1)


