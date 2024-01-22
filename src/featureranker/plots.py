import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import r2_score
from scipy import stats


def plot_correlations(predictions, labels, model_name, save=False):
    plt.figure(figsize=(10, 6))
    plt.scatter(labels, predictions, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs. true values for {model_name}')
    m, b = np.polyfit(labels, predictions, 1)
    plt.plot(labels, m*labels + b, color='red')
    pearson_corr, pearson_pval = stats.pearsonr(labels, predictions)
    spearman_corr, spearman_pval = stats.spearmanr(labels, predictions)
    r2 = r2_score(labels, predictions)
    plt.annotate(f'Pearson: {pearson_corr:.2f} (p={pearson_pval:.2e})', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top')
    plt.annotate(f'Spearman: {spearman_corr:.2f} (p={spearman_pval:.2e})', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=10, verticalalignment='top')
    plt.annotate(f'R2: {r2:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, verticalalignment='top')
    if save:
        title = model_name.replace(' ', '_')
        plt.savefig(f'{title}.png', bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    plt.close()


def plot_confusion_matrix(c_matrix, labels, title='example', save=False):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = c_matrix.max() / 2.
    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        plt.text(j, i, format(c_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if c_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if save:
        title = title.replace(' ', '_')
        plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_after_vote(scoring, title='example', save=False, height_per_feature=0.25, highlight_feature=None):
    features = [item[0] for item in scoring]
    scores = [item[1] for item in scoring]
    fig_height = len(features) * height_per_feature
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.patch.set_facecolor('white')
    colors = ['blue' if feature != highlight_feature else 'yellow' for feature in features]
    ax.barh(features, scores, color=colors, alpha=0.6)
    ax.invert_yaxis()
    label_opts = {'color': 'black', 'bbox': dict(facecolor='white', edgecolor='none')}
    ax.set_xlabel('Scores', **label_opts)
    ax.set_ylabel('Features', **label_opts)
    ax.set_title(f'{title}')
    ax.tick_params(axis='both', which='both', labelsize='large', labelcolor='black', colors='black')
    if save:
        title = title.replace(' ', '_')
        plt.savefig(f'{title}.png', bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    plt.close()


def plot_rankings(rankings, title='example', save=False, height_per_feature=0.25):
    fig, ax = plt.subplots(figsize=(10, len(rankings[0][1]) * height_per_feature))
    colors = ['blue', 'green', 'red', 'yellow', 'cyan']
    legend_labels = []  # List to hold the legend labels

    for i in range(len(rankings)):
        ranking_name, ranking = rankings[i]
        features = ranking[ranking_name].tolist()
        scores = list(reversed(list(range(1, len(features) + 1))))
        ax.barh(features, scores, color=colors[i % len(colors)], alpha=0.3, label=ranking_name, edgecolor='black')
        legend_labels.append(ranking_name)  # Add ranking name to the legend labels

    ax.invert_yaxis()
    label_opts = {'color': 'black', 'bbox': dict(facecolor='white', edgecolor='none')}
    ax.set_xlabel('Scores', **label_opts)
    ax.set_ylabel('Features', **label_opts)
    ax.set_title(f'{title}')
    ax.tick_params(axis='both', which='both', labelsize='large', labelcolor='black', colors='black')
    
    # Add a legend to the plot to show the ranking names and their associated colors
    ax.legend(title='Rankings', labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save:
        title = title.replace(' ', '_')
        plt.savefig(f'{title}.png', bbox_inches='tight', transparent=False, dpi=300)
    
    plt.show()
    plt.close()
