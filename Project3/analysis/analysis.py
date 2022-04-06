import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted

# Import reward_log

def plot_reward_log(args):
    
    reward_log_data = pd.DataFrame()

    for f in natsorted( glob.glob( os.path.join(args.input_folder, 'reward_log*') ) ):
        if reward_log_data.empty:
            reward_log_data = pd.read_csv(f,header=0)
        else:
            pd.concat([reward_log_data, pd.read_csv(f,header=0)], ignore_index=True)

    fig = plt.figure()
    fig.set_size_inches(14, 9)
    plt.plot(reward_log_data)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.grid(which='major', linestyle='-')
    plt.savefig( os.path.join(args.input_folder, 'fig_reward_log.png'), bbox_inches='tight' )

def plot_30_avg_reward(args):
    
    f = glob.glob( os.path.join(args.input_folder, 'analytics*') )[0]    
    analytics_30 = pd.read_csv(f,header=0)
    x = list(range(1, len(analytics_30)*30, 30))
    
    fig = plt.figure()
    fig.set_size_inches(14, 9)
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