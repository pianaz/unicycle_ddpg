import numpy as np
from ddpg import Agent
from utils import plot_learning_curves, save_data
from env import UnicycleEnv

if __name__ == '__main__':
    env = UnicycleEnv()
    agent = Agent(input_dims=env.observation_space.shape, alpha=0.001,
                  beta=0.01, env=env, gamma=0.99,
                  n_actions=env.action_space.shape[0],
                  buffer_size=100000, qfun_layers=[400, 300],
                  policy_layers=[400, 300], batch_size=100, noise=0.05)

    n_episodes = 1000

    figure_file = 'plots/Unicycle'
    data_file = 'plots/Unicycle_data.csv'

    best_score = 0.75
    score_history = []
    goal_history = []
    optimal_history = []
    length_history = [0] * n_episodes
    norm_history=[]
    total_steps = 0
    load_checkpoint = False
    action_noise = True
    training = True

    if load_checkpoint:
        agent.load_models()
       
    for i in range(n_episodes):
        observation = env.reset()
        #states = np.array([observation[:2]])
        optimal = env.shortest_path_length(observation)
        optimal_history.append(optimal)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, action_noise)
            observation_, cost, done, dist, info = env.step(action)
            length_history[i] += dist
            score += cost
            agent.remember(observation, action, cost, observation_, done)
            if training:
                agent.learn()
            observation = observation_
            #states = np.append(states, [observation[:2]], axis=0)
            #env.render()
            total_steps += 1
        
        # to plot trajectories
        #env.render(states)

        if cost<0:
            goal = 'GOAL'
            goal_history.append(1)
            norm_history.append(optimal/length_history[i])
        else:
            goal = 'NO GOAL'
            goal_history.append(0)
            norm_history.append(0)

        score_history.append(score)
        avg_norm = np.mean(norm_history[-100:])
        avg_score = np.mean(score_history[-100:])
        avg_goals = np.sum(goal_history[-100:])

        if avg_norm > best_score and i>=100 and training:
            best_score = avg_norm
            agent.save_models()

        print('episode', i, ' score %.0f ' % score, goal, avg_goals, '/100 '
              ' avg score %.0f ' % avg_score,' norm score %.3f ' % avg_norm,
              ' total steps', total_steps)
  

    if training:
        x = [i+1 for i in range(n_episodes)]
        plot_learning_curves(x, score_history, goal_history, length_history, 
                             optimal_history, figure_file)
        save_data(score_history, goal_history, length_history, optimal_history,
                 data_file)
    x = [i+1 for i in range(n_episodes)]
    plot_learning_curves(x, score_history, goal_history, length_history, 
                            optimal_history, figure_file)