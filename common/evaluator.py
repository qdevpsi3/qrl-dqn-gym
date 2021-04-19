import numpy as np


def evaluate_agent(env, agent, n_eval_episodes):
    episode_steps = []
    episode_reward = []

    for _ in range(n_eval_episodes):
        state = env.reset()
        done = False
        episode_steps.append(0)
        episode_reward.append(0)
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_steps[-1] += 1
            episode_reward[-1] += reward

    episode_steps = np.mean(episode_steps)
    episode_reward = np.mean(episode_reward)
    return {'steps': episode_steps, 'reward': episode_reward}
