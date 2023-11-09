import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(c_matrix, title, labels, save=False):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
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
        plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_ranking(scoring, title, save=False, height_per_feature=0.5):
    for key in list(scoring.keys()):
        features, scores = zip(*scoring[key])
        fig_height = len(features) * height_per_feature
        fig, ax = plt.subplots(figsize=(10, fig_height))
        fig.patch.set_facecolor('white')
        ax.barh(features, scores, color='blue', alpha=0.6)
        ax.invert_yaxis()
        label_opts = {'color': 'black', 'bbox': dict(facecolor='white', edgecolor='none')}
        ax.set_xlabel('Scores', **label_opts)
        ax.set_ylabel('Features', **label_opts)
        ax.set_title(f'{title}')
        ax.tick_params(axis='both', which='both', labelsize='large', labelcolor='black', colors='black')
        if save:
            plt.savefig(f'{title}.png', bbox_inches='tight', transparent=False, dpi=300)
        plt.show()