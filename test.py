from src.featureranker.utils import *
from src.featureranker.plots import *
from src.featureranker.rankers import *
from datasets import load_dataset


df = load_dataset('', split='train').to_pandas()
print(df.head())

view_data(df)

columns_to_drop = [
    'game_date',
    'game_datetime',
    'game_date_adjusted',
    'Unnamed: 0',
    'home_team',
    'away_team',
]
for col in df.columns.tolist():
    if 'id' in col.lower():
        columns_to_drop.append(col)

X, y = get_data(df, target='Home_Win', columns_to_drop=columns_to_drop, n_rows=100)
print(X.shape, y.shape)

rankings = feature_ranking(X, y, task='classification', choices=['l1'])
scoring = voting(rankings, save=True, save_path='test.csv')
plot_after_vote(scoring)
