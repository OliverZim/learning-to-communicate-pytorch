"""

DRQN-based agent that learns to communicate with other agents to play 
the Switch game.

"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from modules.bn_rnn import RNN


class SwitchCNet(nn.Module):

	def __init__(self, opt):
		super(SwitchCNet, self).__init__()

		self.opt = opt
		self.comm_size = opt.game_comm_bits
		self.init_param_range = (-0.08, 0.08)

		# Set up inputs
		self.agent_lookup = nn.Embedding(opt.game_nagents, opt.model_rnn_size)
		self.state_lookup = nn.Embedding(2, opt.model_rnn_size)

		# Action aware
		if opt.model_action_aware:
			if opt.model_dial:
				self.prev_action_lookup = nn.Embedding(opt.game_action_space_total, opt.model_rnn_size)
			else:
				self.prev_action_lookup = nn.Embedding(opt.game_action_space + 1, opt.model_rnn_size)
				self.prev_message_lookup = nn.Embedding(opt.game_comm_bits + 1, opt.model_rnn_size)

		# Communication enabled
		if opt.comm_enabled:
			self.messages_mlp = nn.Sequential()
			if opt.model_dial and opt.model_bn:
				self.messages_mlp.add_module('batchnorm1', nn.BatchNorm1d(self.comm_size))
			self.messages_mlp.add_module('linear1', nn.Linear(self.comm_size, opt.model_rnn_size))
			if opt.model_comm_narrow:
				self.messages_mlp.add_module('relu1', nn.ReLU(inplace=True))

		# Set up RNN
		rnn_mode = opt.model_rnn or 'gru'
		dropout_rate = opt.model_rnn_dropout_rate or 0
		self.rnn = RNN(
			mode=rnn_mode, input_size=opt.model_rnn_size, hidden_size=opt.model_rnn_size, 
			num_layers=opt.model_rnn_layers, use_bn=opt.model_bn, bn_max_t=6, dropout_rate=dropout_rate)
		

		# Set up outputs
		self.outputs = nn.Sequential()
		if dropout_rate > 0:
			self.outputs.add_module('dropout1', nn.Dropout(dropout_rate))
		self.outputs.add_module('linear1', nn.Linear(opt.model_rnn_size, opt.model_rnn_size))
		self.outputs.add_module('relu1', nn.ReLU(inplace=True))
		self.outputs.add_module('linear2', nn.Linear(opt.model_rnn_size, opt.game_action_space_total))

		self.reset_params()

	def _reset_linear_module(self, 	layer):
		layer.weight.data.uniform_(*self.init_param_range)
		layer.bias.data.uniform_(*self.init_param_range)

	def reset_params(self):
		opt = self.opt
		self.agent_lookup.weight.data.uniform_(*self.init_param_range)
		self.state_lookup.weight.data.uniform_(*self.init_param_range)
		if opt.model_action_aware:
			self.prev_action_lookup.weight.data.uniform_(*self.init_param_range)
			if not opt.model_dial:
				self.prev_message_lookup.weight.data.uniform_(*self.init_param_range)
		self._reset_linear_module(self.messages_mlp.linear1)
		self.rnn.reset_params()
		self._reset_linear_module(self.outputs.linear1)
		self._reset_linear_module(self.outputs.linear2)

	def get_params(self):
		return list(self.parameters())

	def forward(self, s_t, messages, hidden, prev_action, agent_index):
		opt = self.opt

		s_t = Variable(s_t)
		hidden = Variable(hidden)
		prev_message = None
		if opt.model_dial:
			prev_action = Variable(prev_action)
		else:
			prev_action, prev_message = prev_action
			prev_action = Variable(prev_action)
			prev_message = Variable(prev_message)
			messages = Variable(messages)
		agent_index = Variable(agent_index)

		z_a, z_o, z_u, z_m = [0]*4
		z_a = self.agent_lookup(agent_index)
		z_o = self.state_lookup(s_t)
		if opt.model_action_aware:
			z_u = self.prev_action_lookup(prev_action)
			if prev_message is not None:
				z_u += self.prev_message_lookup(prev_message)

		z_m = self.messages_mlp(messages.view(-1, self.comm_size))

		z = z_a + z_o + z_u + z_m
		z = z.unsqueeze(1)

		rnn_out, _ = self.rnn(z, hidden=hidden)
		outputs = self.outputs(rnn_out[:, -1, :])

		return rnn_out, outputs
