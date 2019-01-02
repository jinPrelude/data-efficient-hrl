import numpy as np

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size, random_seed = np.array([-1])):
		test = np.sum(random_seed)
		if np.sum(random_seed) == -1 :
			random_seed = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in random_seed:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

	def extract(self):
		ind = np.arange(0, len(self.storage))
		x, y, u, r, d = [], [], [], [], []

		for i in ind:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		self.storage = []
		self.ptr = 0
		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

	def reset(self):
		self.storage = []
		self.ptr = []


class Episode_ReplayBuffer(object) :
	def __init__(self, max_size = 1e4):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample_episode(self, batch_size, random_seed = np.array([-1])):
		if random_seed.all() == -1 :
			random_seed = np.random.randint(0, len(self.storage), size=batch_size)
		t = []

		for i in range(random_seed.size):
			T = self.storage[random_seed[i]]
			t.append(T)
		return t

	def sample(self, batch_size):

		t = []
		x, y, u, r, d = [], [], [], [], []

		for i in range(len(self.storage)):
			T = self.storage[i]

			X, Y, U, R, D = T
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		# length = x[0][0].shape[0]		# size of observation + goal

		s, s2, a, re, do = [], [], [], [], []
		for k in range(batch_size) :

			episode_random_num = np.random.randint(0, len(self.storage), 1)[0]
			test = x[episode_random_num].shape[0]
			step_random_num = np.random.randint(0, x[episode_random_num].shape[0], 1)

			s.append(x[episode_random_num][step_random_num][0])
			s2.append(y[episode_random_num][step_random_num][0])
			a.append(u[episode_random_num][step_random_num][0])
			re.append(r[episode_random_num][step_random_num][0])
			do.append(d[episode_random_num][step_random_num][0])


		return np.array(s), np.array(s2), np.array(a), np.array(re).reshape(-1, 1), np.array(do).reshape(-1, 1)
