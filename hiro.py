import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3
import time



# Runs policy for X episodes and returns average reward
def evaluate_policy(policy_high, policy, eval_episodes=5):
    avg_reward = 0.

    for _ in range(eval_episodes) :

        obs = env.reset()
        high_input = np.concatenate((obs['observation'], obs['desired_goal']), axis=0)
        done = False

        while not done :
            goal = policy_high.select_action(np.array(high_input))

            while not done :


                low_input = np.concatenate((obs['observation'], goal), axis=0)
                action = policy.select_action(low_input)
                #print('action : ', action)
                new_obs, reward, done, _ = env.step(action)

                avg_reward += reward

                if -np.sum(abs(obs['observation'] + goal - new_obs['observation'])) > args.reward_threshold :
                    obs = new_obs
                    break

                if args.render :
                    env.render()

                obs = new_obs

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

def re_labeling(policy, replay_buffer_low, replay_buffer_high, relabel_replay_buffer, batch_size, state_dim) :

    # Sample replay buffer
    random_seed = np.random.randint(0, len(replay_buffer_high.storage), size=batch_size)
    X = replay_buffer_low.sample_episode(len(replay_buffer_low.storage), random_seed)
    X_high, Y_high, _, r, d = replay_buffer_high.sample(len(replay_buffer_high.storage), random_seed)

    # first episode
    g = []

    # reset relabel replay buffer
    relabel_replay_buffer.reset()

    # select specific episode
    for i in range(len(X)) :
        tmp_state, tmp_action, tmp_next_state, tmp_done, tmp_reward = [], [], [], [], []

        # gather steps in selected episode
        for j in range(X[i][0].shape[0]) :
            """
            state[j] = torch.FloatTensor(X[i, j, 0])
            action[j] = torch.FloatTensor(X[i, j, 1])
            next_state[j] = torch.FloatTensor(X[i, j, 2])
            done[j] = torch.FloatTensor(1 - X[i, j, 3])
            reward[j] = torch.FloatTensor(X[i, j, 4])
            """

            tmp_state.append(X[i][0][j])
            tmp_next_state.append(X[i][1][j])
            tmp_action.append(X[i][2][j])
            tmp_done.append((1 - X[i][3][j]))
            tmp_reward.append(X[i][4][j])

        state = np.asarray(tmp_state)
        action = np.asarray(tmp_action)
        next_state = np.asarray(tmp_next_state)
        done = np.asarray(tmp_done)
        reward = np.asarray(tmp_reward)

        # make virtual goals
        candidate = []
        candidate_score = []
        candidate.append(state[0, state_dim:]) # original goal
        candidate.append(np.asarray(state[-1, :state_dim]) - np.asarray(state[0, :state_dim]))    # s_(t+c) - s_t
        for _ in range(8) :
            candidate.append(np.random.normal(candidate[1], 1.0))
        candidate = np.asarray(candidate)

        # estimate each goal_bar candidate
        for k in range(10) :
            virtual_goal = candidate[k]
            score = 0
            # episode iteration
            for l in range(0, state.shape[0], 2) :
                score -= np.sum(abs(action[l] - policy.select_action(np.concatenate((state[l, :state_dim], virtual_goal), axis=0)))/2)

                # virtual goal update
                if l >= (state.shape[0]-1) :
                    continue
                else :
                    virtual_goal = state[l, state_dim] + virtual_goal - state[l+1, :state_dim]
            candidate_score.append(score)

        max_num = np.argmax(candidate_score)
        g.append(candidate[max_num])

        relabel_replay_buffer.add((X_high[i], Y_high[i], g[i], r[i], d[i]))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="FetchReach-v1")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_episode", default=200,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--high_start_episode", default=700,
                        type=int)  # How many time steps purely random policy is run for

    parser.add_argument("--eval_freq", default=10, type=int)  # How often (episode) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=40, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--reward_threshold", default=-5, type=float) # low_policy reward threshold
    parser.add_argument("--render", default=True, type=bool)  # low_policy reward threshold

    # HIRO parameters
    parser.add_argument("--high_train_episode", default=400, type=int)


    args = parser.parse_args()

    # Generate model_save directory
    file_name = "%s_%s_%s" % ('policy_low', args.env_name, str(args.seed))
    file_name_high = "%s_%s_%s" % ('policy_high', args.env_name, str(args.seed))

    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Init environments
    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.spaces['observation'].shape[0]
    action_dim = env.action_space.shape[0]
    env_goal_dim = env.observation_space.spaces['desired_goal'].shape[0]
    goal_dim = state_dim
    max_action = env.action_space.high[0]
    max_state = 200.

    # Initialize policy
    policy_high = TD3.TD3(state_dim+env_goal_dim, goal_dim, max_state)
    policy = TD3.TD3(state_dim*2, action_dim, max_action)


    episode_replay_buffer = utils.ReplayBuffer()
    replay_buffer_low = utils.Episode_ReplayBuffer()
    replay_buffer_high = utils.ReplayBuffer()
    relabel_replay_buffer = utils.ReplayBuffer()


    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy_high, policy)]

    total_timesteps = 0
    episode_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:


        if done:

                # Evaluate episode
            if episode_since_eval >= args.eval_freq:
                episode_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy_high, policy))

                if args.save_models: policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_since_eval += 1

        low_reward_sum = 0
        high_reward = 0
        low_goal_reach = False
        high_input = np.concatenate((obs['observation'], obs['desired_goal']), axis=0)

        # Select action randomly or according to policy
        if episode_num < args.high_start_episode:
            goal = np.random.uniform(-200., 200., int(env.observation_space.spaces['observation'].shape[0]))
        else:
            goal = policy_high.select_action(high_input)
            if args.expl_noise != 0:
                goal = (goal + np.random.normal(0, args.expl_noise,
                                                                  size=env.observation_space.spaces['observation'].shape[0])).clip(
                    -max_state, max_state)

        # Low_Policy start
        while total_timesteps < args.max_timesteps:

            if done:

                # Store data in replay buffer
                replay_buffer_high.add((high_input, np.concatenate((new_obs['observation'], new_obs['desired_goal']), axis=0), goal, high_reward, done_bool))

                # add episode to low replay buffer
                replay_buffer_low.add(episode_replay_buffer.extract())

                if total_timesteps != 0:


                    print("Total T : ", total_timesteps, "  Episode Num : ", episode_num, "  Episode T : ",
                          episode_timesteps, "  Reward : ", episode_reward, "  Low reward : ", low_reward_sum)
                    policy.train(replay_buffer_low, episode_timesteps, args.batch_size, args.discount, args.tau,
                                  args.policy_noise, args.noise_clip, args.policy_freq)

                    if episode_num > args.high_train_episode :
                        print('high_policy_training..')
                        #start_time = time.time()
                        re_labeling(policy, replay_buffer_low, replay_buffer_high, relabel_replay_buffer,
                                    args.batch_size, state_dim)
                        #print(time.time() - start_time)

                        policy_high.train(relabel_replay_buffer, int(episode_timesteps / 10), args.batch_size,
                                          args.discount, args.tau,
                                          args.policy_noise, args.noise_clip, args.policy_freq)

                break

            # Select action randomly or according to policy

            low_input = np.concatenate((obs['observation'], goal), axis=0)

            if episode_num < args.start_episode:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(low_input))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)

            if args.render:
                env.render()

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            episode_reward += reward
            high_reward += reward

            low_reward = -abs(np.sum(obs['observation'] + goal - new_obs['observation']))
            goal = obs['observation'] + goal - new_obs['observation']
            low_reward_sum += low_reward


            episode_replay_buffer.add((low_input, np.concatenate((new_obs['observation'], goal), axis=0), action, low_reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1

            if low_reward > args.reward_threshold :
                print('goal reached')
                low_goal_reach = True
                break




    # Final evaluation
    evaluations.append(evaluate_policy(policy_high, policy))
    if args.save_models:
        policy.save("%s" % (file_name), directory="./pytorch_models")
        policy_high.save("%s" % (file_name_high), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)

