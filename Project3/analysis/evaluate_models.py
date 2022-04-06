import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from environment import Environment
import numpy as np
import argparse
from agent_dqn import Agent_DQN

def run(args):
    if args.test_dqn: # this is mostly to ensure that the variable exist so that it will be used in AgentDQN
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)

        pth_files = glob.glob( os.path.join( '*pth') )
        unsorted_pth_datetimes = [os.path.splitext(i)[0][-11:] for i in pth_files]

        data_rows = []

        # Iterate through all the models
        for ind in np.argsort(unsorted_pth_datetimes):
            model_file_path = glob.glob( os.path.join( '*pth') ) [ind]
            args.test_model_file_path = model_file_path
            agent = Agent_DQN(env, args)
            
            # Store rewards for current model name
            reward = obtain_rewards(agent, env, total_episodes=100)

            data_rows.append( [model_file_path, reward] )

        output_df = pd.DataFrame(data_rows, columns=['model_filename', 'score'])
        output_df.to_csv('complete_results.csv', index=False)

def obtain_rewards(agent, env, total_episodes=100):
    rewards = []
    env.seed(11037) # hardcoded by assignment
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return np.mean(rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--model_directory', type=str, help='path to model files.')

    args = parser.parse_args()
    
    # Manually set the argument values
    args.test_dqn = True
    args.logging_enabled = False
    args.config = None

    # Ideally this code is in an upper level, so it's not stored with the model files
    # Change working directory
    os.chdir(args.model_directory)

    run(args)