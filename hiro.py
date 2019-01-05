import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3
import time



# Runs policy for X episodes and returns average reward
# 성능 테스트를 위한 함수입니다!
def evaluate_policy(policy_high, policy, eval_episodes=5):
    avg_reward = 0.
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

                if -np.sum(abs(obs['observation'] + goal - new_obs['observation'])) > args.reward_threshold :
                    print('goal reached')
                    obs = new_obs
                    break

                if args.render :
                    env.render()

                obs = new_obs

    avg_reward /= eval_episodes

    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

# 리라벨링을 해주고 리라벨링 된 메모리를 relabel_replay_buffer에 저장해줍니다.
def re_labeling(policy, replay_buffer_low, replay_buffer_high, relabel_replay_buffer, batch_size, state_dim) :

    # Sample replay buffer
    # replay_buffer_high 의 보유 메모리 갯수 내에서 batch_size 만큼의 숫자를 랜덤 선정합니다.
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

            tmp_state.append(X[i][0][j])
            tmp_next_state.append(X[i][1][j])
            tmp_action.append(X[i][2][j])
            tmp_done.append((1 - X[i][3][j]))
            tmp_reward.append(X[i][4][j])

        state = np.asarray(tmp_state)
        action = np.asarray(tmp_action)

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
            for l in range(0, state.shape[0], 1) :
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

    # 하이퍼파라메터 세팅 부분입니다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="FetchReach-v1")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds

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
    parser.add_argument("--render", default=False, type=bool)  # low_policy reward threshold

    # train start  hyperparameters
    # 학습 시작 시점과 관련된 파라메터 값들입니다.
    parser.add_argument("--high_train_episode", default=80, type=int)
    parser.add_argument("--start_episode", default=30,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--high_start_episode", default=50,
                        type=int)  # How many time steps purely random policy is run for

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
    # 환경을 생성합니다.
    env = gym.make(args.env_name)

    # Set seeds
    # 시드를 넣어줍니다
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 환경 정보들을 세팅해줍니다. 이 정보를 policy 한테 줘야 하거든요
    state_dim = env.observation_space.spaces['observation'].shape[0]
    action_dim = env.action_space.shape[0]
    env_goal_dim = env.observation_space.spaces['desired_goal'].shape[0]
    goal_dim = state_dim
    max_action = env.action_space.high[0]
    max_state = 200.

    # Initialize policy
    # policy 를 초기화 생성해줍니다.
    policy_high = TD3.TD3(state_dim+env_goal_dim, goal_dim, max_state)
    policy = TD3.TD3(state_dim*2, action_dim, max_action)

    # replay buffer 들을 만들어줍니다.
    episode_replay_buffer = utils.ReplayBuffer() # 에피소드씩만을 저장합니다. 에피소드가 끝난 후 replay_buffer_low에 extract 함수로 데이터를 넘겨준 다음 자동으로 초기화됩니다.
    replay_buffer_low = utils.Episode_ReplayBuffer() # low_polcy를 위한 메모리입니다. 스텝단위가 아닌 에피소드 단위로 저장합니다.
    replay_buffer_high = utils.ReplayBuffer()   # high_policy를 위한 메모리입니다.
    relabel_replay_buffer = utils.ReplayBuffer() # re_label 만을 위한 메모리입니다. re_labeling 함수가 시작되면 초기화되고 학습을 위한 re-labeling이 된 메모리를 저장하여 high_policy 학습에 쓰입니다.



    # Evaluate untrained policy
    # 학습되기 전 정책을 평가합니다.
    evaluations = [evaluate_policy(policy_high, policy)]

    total_timesteps = 0
    episode_since_eval = 0
    episode_num = 0
    done = True
    low_goal_reach = False

    while total_timesteps < args.max_timesteps:


        if done or low_goal_reach:

            # low_goal_reach로 해당 반복문 진입에 성공하였기 때문에 다시 값을 False 로 바꿔줍니다.
            if low_goal_reach:
                low_goal_reach = False

            else :
                # Evaluate episode
                # 정책을 평가합니다.
                if episode_since_eval >= args.eval_freq:    # 일정한 간격을 두고 evaluate를 실행합니다.
                    episode_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(policy_high, policy))

                    if args.save_models: policy.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)

                # Reset environment
                # 환경을 초기화시켜줍니다.
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                episode_since_eval += 1

        # 처음 코드를 실행시키면 무조건 done에 들어와서 파라메터를 초기화해주고 시작합니다. 그 다움부터는 에피소드가 끝날대마다 초기화됩니다.
        low_reward_sum = 0
        high_reward = 0
        low_goal_reach = False
        high_input = np.concatenate((obs['observation'], obs['desired_goal']), axis=0)

        # Select action randomly or according to policy
        # 일정 에피소드 전까지는 랜덤하게 gaol이 초기화되고, 그 이후부터는 노이즈를 더한 high_policy의 출력이 goal이 됩니다.
        if episode_num < args.high_start_episode:
            goal = np.random.uniform(-max_state, max_state, int(env.observation_space.spaces['observation'].shape[0]))
        else:
            goal = policy_high.select_action(high_input)
            if args.expl_noise != 0:
                goal = (goal + np.random.normal(0, args.expl_noise,
                                                size=env.observation_space.spaces['observation'].shape[0])).clip(-max_state, max_state)

        # Low_Policy start
        # low_policy가 시작됩니다.
        while total_timesteps < args.max_timesteps:

            # done 이나 low_goal_reach가 True 먼저 판단한 후 실행 코드로 넘어갑니다.
            if done or low_goal_reach:

                # 코드 실행 처음에 실행되는 것을 방지하기 위한 조치
                if total_timesteps != 0:

                    # Store data in replay buffer
                    # 에피소드가 끝났으니 high_policy 메모리에 저장해줍니다.
                    replay_buffer_high.add((high_input,
                                            np.concatenate((new_obs['observation'], new_obs['desired_goal']), axis=0),
                                            goal, high_reward, done_bool))

                    # add episode to low replay buffer
                    # episode_replay_buffer에 저장되어있는 메모리를 가저와 저장합니다. 동시에 episode_replay_buffer은 초기화됩니다.
                    replay_buffer_low.add(episode_replay_buffer.extract())

                    # 에피소드 결과와 경과를 출력해줍니다.
                    print("Total T : ", total_timesteps, "  Episode Num : ", episode_num, "  Episode T : ",
                          episode_timesteps, "  Reward : ", episode_reward, "  Low reward : ", low_reward_sum)

                    # low_policy를 학습시켜줍니다.
                    policy.train(replay_buffer_low, episode_timesteps, args.batch_size, args.discount, args.tau,
                                  args.policy_noise, args.noise_clip, args.policy_freq)


                    # high policy는 메모리가 늦게 쌓이므로 일정 에피소드 뒤부터 학습을 시켜줍니다.
                    if episode_num > args.high_train_episode :
                        #start_time = time.time()
                        # replay_buffer_low 와 replay_buffer_high 를 받아 re_labeling을 해주고 relabel_replay_buffer에 저장해줍니다.
                        re_labeling(policy, replay_buffer_low, replay_buffer_high, relabel_replay_buffer,
                                    args.batch_size, state_dim)
                        #print(time.time() - start_time)

                        # re_labeling 된 메모리를 이용하여 high_policy를 트래이닝 시켜줍니다.
                        policy_high.train(relabel_replay_buffer, int(episode_timesteps / 10), args.batch_size,
                                          args.discount, args.tau,
                                          args.policy_noise, args.noise_clip, args.policy_freq)


                # high_policy 반복문에게 끝났다는 사실을 알리기 위해 반복문에서 나와줍니다.
                break


            # low 의 인풋으로 들어가도록 observation과 high_policy의 goal을 합쳐줍니다.
            low_input = np.concatenate((obs['observation'], goal), axis=0)

            # Select action randomly or according to policy
            # 일정 에피소드 전까지는 랜덤하게 action이 초기화되고, 그 이후부터는 노이즈를 더한 low_policy의 출력이 action이 됩니다.
            if episode_num < args.start_episode:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(low_input))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)

            # 파라메터 세팅값에 따라 렌더링을 해줍니다.
            if args.render:
                env.render()

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            episode_reward += reward
            high_reward += reward

            # low reward 계산을 해주고 goal을 재설정합니다.
            low_reward = -abs(np.sum(obs['observation'] + goal - new_obs['observation']))
            goal = obs['observation'] + goal - new_obs['observation']
            low_reward_sum += low_reward

            # episode_replay_buffer에 저장해줍니다.
            episode_replay_buffer.add((low_input, np.concatenate((new_obs['observation'], goal), axis=0), action, low_reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1

            # goal 이 달성되었다면 (우리가 설정한 범위 안에 들어오게 된다면) low_goal_reach를 True 로 바꿔줍니다. done 조건문에 잡힐것입니다.
            if low_reward > args.reward_threshold :
                print('goal reached')
                low_goal_reach = True





    # Final evaluation
    evaluations.append(evaluate_policy(policy_high, policy))
    if args.save_models:
        policy.save("%s" % (file_name), directory="./pytorch_models")
        policy_high.save("%s" % (file_name_high), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)

