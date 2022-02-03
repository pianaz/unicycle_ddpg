import numpy as np
import csv
import matplotlib.pyplot as plt

def plot_learning_curves(x, scores, goals, lengths, optim, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file + '_score.pdf')
    plt.close()
    for i in range(len(running_avg)):
        running_avg[i] = np.sum(goals[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.xlabel('Episodes')
    plt.ylabel('Goals')
    plt.title('Running average of previous 100 goals')
    plt.savefig(figure_file + '_goals.pdf')
    plt.close()
    optim = np.array(optim)
    lengths = np.array(lengths)
    ratio = optim/lengths
    ratio[np.array(goals)==0] = 0
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(ratio[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.xlabel('Episodes')
    plt.ylabel('Normalized score')
    plt.yticks([0,1])
    plt.title('Running average of normalized cost')
    plt.savefig(figure_file + '_normalized_score.pdf')
    plt.close()

def save_data(scores, goals, lengths, opts, data_file):
    file = open(data_file, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['score', 'avg_score', 'goal', 'avg_goals', 'path_length',
                    'optimal_length', 'normalized_score'])
    for w in range(len(scores)):
        writer.writerow([scores[w], np.mean(scores[max(0, w-100):(w+1)]),
                    goals[w], np.sum(goals[max(0, w-100):(w+1)]), lengths[w],
                    opts[w], opts[w]/lengths[w] if goals[w] else 0])
    file.close()