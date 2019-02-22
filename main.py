import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3

def low_reward_cal(obs, new_obs, goal) :
	return -abs(np.sum(obs + goal - new_obs))

def sub_goal_transition(obs, new_obs, goal) :
	return obs+goal-new_obs

def evaluate_policy(policy, policy_high, eval_episodes=1):
	avg_reward = 0.
	low_reward = 0.
	print("---------------------------------------")



	for _ in range(eval_episodes) :

		obs = env.reset()
		high_input = np.concatenate((obs['observation'], obs['desired_goal']), axis=0)
		done = False

		while not done :
			goal = policy_high.select_action(np.array(high_input))

			while not done :

				low_input = np.concatenate((obs['observation'], goal), axis=0)
				action = policy.select_action(low_input)
				new_obs, reward, done, _ = env.step(action)

				avg_reward += reward
				low_reward += low_reward_cal(obs['observation'], new_obs['observation'], goal)

				if -np.sum(abs(obs['observation'] + goal - new_obs['observation'])) > args.reward_threshold :
					print('goal reached')
					obs = new_obs
					break

				if args.render :
					env.render()

				obs = new_obs
				goal = sub_goal_transition(obs['observation'], new_obs['observation'], goal)

	avg_reward /= eval_episodes
	low_reward /= eval_episodes
	print("Evaluation over %d episodes: %.3f, low_rewrad : %.3f" % (eval_episodes, avg_reward, low_reward))
	print("---------------------------------------")
	return avg_reward



def low_train(low_policy, replay_buffer, episode_timesteps, batch_size, discount, tau,
			  policy_noise, noise_clip, policy_freq) :

	tmp_memory = utils.Normal_ReplayBuffer()

	x, _, g, y, _, n_g, u, _, d, r = replay_buffer.step_sample(batch_size)
	state = torch.FloatTensor(x)
	sub_goal = torch.FloatTensor(g)
	next_state = torch.FloatTensor(y)
	next_sub_goal = torch.FloatTensor(n_g)
	action = torch.FloatTensor(u)
	reward = torch.FloatTensor(r)
	done = torch.FloatTensor(1 - d)


	for i in range(batch_size) :

		obs = np.concatenate((state[i], sub_goal[i]), axis=0)
		next_obs = np.concatenate((next_state[i], next_sub_goal[i]), axis=0)

		tmp_memory.add((obs, next_obs, action[i], done[i], reward[i]))

	low_policy.train(tmp_memory, episode_timesteps, batch_size, discount, tau,
					 policy_noise, noise_clip, policy_freq)


def high_train(high_policy, low_policy, replay_buffer, episode_timesteps, batch_size, discount, tau,
			   policy_noise, noise_clip, policy_freq, env_goal) :

	tmp_memory = utils.Normal_ReplayBuffer()

	t = replay_buffer.episode_sample(batch_size)

	# goal re_labeling
	for i in range(batch_size) :
		state, env_goal, sub_goal, next_state, next_env_goal, next_sub_goal, action, reward = [], [], [], [], [], [], [], []
		for j in range(args.c_step) :
			state.append(torch.FloatTensor(t[i][1][0]))
			env_goal.append(torch.FloatTensor(t[i][1][1]))
			sub_goal.append(torch.FloatTensor(t[i][1][2]))
			next_state.append(torch.FloatTensor(t[i][1][3]))
			next_env_goal.append(torch.FloatTensor(t[i][1][4]))
			next_sub_goal.append(torch.FloatTensor(t[i][1][5]))
			action.append(torch.FloatTensor(t[i][1][6]))
			reward.append(torch.FloatTensor([t[i][1][7]]))

		candidate = []
		candidate_score = []
		candidate.append(sub_goal[0])
		candidate.append(next_state[-1]-state[0])
		for _ in range(8) :
			candidate.append(np.random.normal(candidate[1], args.max_observation/2)) # 주석 : candidate 생성 분포 범위 변수로 만들어라 - 완료
		candidate = np.asarray(candidate)

		for virtual_goal in candidate :
			score = 0
			for j in range(args.c_step) :
				a = np.array(action[j])
				mu = low_policy.select_action(np.concatenate((state[j], virtual_goal), axis=0))
				score -= abs(np.sum(a-mu))

				virtual_goal = sub_goal_transition(state[j], next_state[j], torch.FloatTensor(virtual_goal))
			candidate_score.append(score)

		num = np.argmax(candidate_score)
		tmp_memory.add((np.concatenate((state[0],  env_goal[0]), axis=0), np.concatenate((next_state[-1], next_env_goal[-1]),axis=0), candidate[num], np.sum(np.array(reward)), 0.0))


	high_policy.train(tmp_memory, episode_timesteps, int(batch_size/5), discount, tau,
					 policy_noise, noise_clip, policy_freq)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="TD3")					# Policy name
	parser.add_argument("--env_name", default="FetchReach-v1")			# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--high_start_timesteps", default=5e2, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--low_start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=5e2, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=3e7, type=float)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=50, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--render", default=False, type=bool)			# Render

	# hrl parameters
	parser.add_argument("--max_observation", default=200., type=float)
	parser.add_argument("--c_step", default=50, type=int)
	parser.add_argument("--reward_threshold", default=-1, type=float)
	parser.add_argument("--high_train_start", default=200, type=int)


	args = parser.parse_args()

	low_file_name = "%s_%s_%s_low" % ('TD3', args.env_name, str(args.seed))
	high_file_name = "%s_%s_%s_high" % ('TD3', args.env_name, str(args.seed))
	print ("---------------------------------------")
	print ("Settings: %s %s" % (low_file_name, high_file_name))
	print ("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if args.save_models and not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")

	env = gym.make(args.env_name)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.spaces['observation'].shape[0]
	goal_dim = env.observation_space.spaces['desired_goal'].shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Initialize policy
	low_policy = TD3.TD3(state_dim+state_dim, action_dim, max_action, lr = 0.02)
	high_policy = TD3.TD3(state_dim+goal_dim, state_dim, args.max_observation, lr=0.003)
	replay_buffer = utils.ReplayBuffer(args.c_step)
	
	# Evaluate untrained policy
	evaluations = [evaluate_policy(low_policy, high_policy)]

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	done = True 

	while total_timesteps < args.max_timesteps:

		if total_timesteps % args.c_step == 0 :

			if total_timesteps != 0 :

				replay_buffer.episode_add()

				low_train(low_policy, replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
						  args.policy_noise, args.noise_clip, args.policy_freq)
				if episode_num > args.high_train_start :
					high_train(high_policy, low_policy, replay_buffer, int(episode_timesteps/10), args.batch_size, args.discount, args.tau,
								  	args.policy_noise, args.noise_clip, args.policy_freq, obs['desired_goal'])

				print('train..', episode_num)

				if total_timesteps % args.eval_freq == 0 :
					evaluate_policy(low_policy, high_policy, 5)

			# Reset environment
			obs = env.reset()
			done = False
			c_step_reward = 0
			episode_reward = 0
			episode_low_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# generate input
		high_input = np.concatenate((obs['observation'], obs['desired_goal']), 0)
		# Select action randomly or according to policy
		if total_timesteps < args.high_start_timesteps:
			# action = env.action_space.sample()
			sub_goal = np.random.uniform(-args.max_observation, args.max_observation, state_dim)
		else:
			sub_goal = high_policy.select_action(high_input)
			if args.expl_noise != 0:
				sub_goal = (sub_goal + np.random.normal(0, args.expl_noise, size=env.observation_space.spaces['observation'].shape[0])).clip(-args.max_observation, args.max_observation)

		# low policy 부분 들어가야함
		while True :

			# generate low_input
			low_input = np.concatenate((obs['observation'], sub_goal), 0)


			if total_timesteps < args.low_start_timesteps:
				# action = env.action_space.sample()
				action = env.action_space.sample()
			else:
				action = low_policy.select_action(low_input)
				if args.expl_noise != 0:\
					action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
						env.action_space.low, env.action_space.high)

			# Perform action
			new_obs, reward, done, _ = env.step(action)
			if args.render :
				env.render()
			done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)


			low_reward = low_reward_cal(obs['observation'], new_obs['observation'], sub_goal)
			episode_low_reward += low_reward
			episode_reward += reward
			c_step_reward += reward


			new_sub_goal = sub_goal_transition(obs['observation'], new_obs['observation'], sub_goal)


			replay_buffer.step_add((obs['observation'], obs['desired_goal'], sub_goal, new_obs['observation'], new_obs['desired_goal'], new_sub_goal, action, float(reward), done_bool, float(low_reward)))


			obs = new_obs
			sub_goal = new_sub_goal

			episode_timesteps += 1
			total_timesteps += 1
			timesteps_since_eval += 1

			if low_reward > args.reward_threshold :		# 주석 : low_reward threshold 를 가변번수로 바꿔주시오
				break
			if done :
				break
			if total_timesteps % args.c_step == 0 :
				break

		if total_timesteps % args.c_step == 0 :
			continue

		elif low_reward > args.reward_threshold : # 주석 : low_reward threshold 를 가변번수로 바꿔주시오
			print('low_goal reached')
			continue


		# 에피소드가 끝났을 때 : low_policy
		elif done:

			if total_timesteps != 0:
				# print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
				print("Total T : ", total_timesteps, "	Episode Num : ", episode_num, "  Episode T : ",
					  episode_timesteps, "  Reward : ", episode_reward)
				'''
				low_policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
								 args.policy_noise, args.noise_clip, args.policy_freq)
				'''

			'''
			# Evaluate episode
			if timesteps_since_eval >= args.eval_freq:
				timesteps_since_eval %= args.eval_freq
				evaluations.append(evaluate_policy(low_policy))

				if args.save_models: low_policy.save(file_name, directory="./pytorch_models")
				np.save("./results/%s" % (file_name), evaluations)


			# Store data in replay buffer
			'''
			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_low_reward = 0
			episode_timesteps = 0
			episode_num += 1

			# 주석 : 이 메서드 확인 끝났음 저장하는거 들어가셈


	# Final evaluation 
	evaluations.append(evaluate_policy(low_policy, high_policy))
	if args.save_models:
		low_policy.save("%s" % (low_file_name), directory="./pytorch_models")
		high_policy.save("%s" % (high_file_name), directory="./pytorch_models")
	np.save("./results/%s_%s_%s" % ('TD3', args.env_name, args.seed), evaluations)
