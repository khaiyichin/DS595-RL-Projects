import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
import numpy as np

# Import reward_log

def plot_reward_log(args):
    
    reward_log_data = [] # for some fucking reason pandas concat is being a fucking bitch adding fucking nan fucking columns

    for f in natsorted( glob.glob( os.path.join(args.input_folder, 'reward_log*') ) ):
        if len(reward_log_data) == 0:
            temp = pd.read_csv(f,header=0)
            reward_log_data = temp.to_numpy()
        else:
            temp = pd.read_csv(f,header=0)
            reward_log_data = np.vstack( (reward_log_data,temp.to_numpy()) )

    x = list(range(1, len(reward_log_data)*5, 5))

    fig = plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot(x, reward_log_data)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.grid(which='major', linestyle='-')
    plt.savefig( os.path.join(args.input_folder, 'fig_reward_log.png'), bbox_inches='tight' )

def plot_30_avg_reward(args):
    
    f = glob.glob( os.path.join(args.input_folder, 'analytics*') )[0]    
    analytics_30 = pd.read_csv(f,header=0)
    x = list(range(1, len(analytics_30)*30*5, 30*5))
    
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot( x, analytics_30.iloc[:,0] )
    plt.xlabel('Episodes')
    plt.ylabel('Average score')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.grid(which='major', linestyle='-')
    plt.savefig( os.path.join(args.input_folder, 'fig_analytics_30.png'), bbox_inches='tight' )

# Plot results
# Depending if plotting using analytics.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--input_folder', type=str, help='path to files.')
    parser.add_argument('--plot_reward_log', action="store_true")
    parser.add_argument('--plot_30_avg_reward', action="store_true")

    args = parser.parse_args()

    if args.plot_reward_log:
        plot_reward_log(args)

    if args.plot_30_avg_reward:
        plot_30_avg_reward(args)