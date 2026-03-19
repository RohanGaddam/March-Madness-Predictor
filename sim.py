#%%
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.feature_selection import SelectKBest
#%%
data_2023 = pd.read_csv('Data/2023_stats/2023_data.csv')
ken_2023 = pd.read_csv('Data/2023_stats/2023_Kenpom.csv')
res_2023 = pd.read_csv('Data/2023_stats/2023_results.csv')
#%%
data_2024 = pd.read_csv('Data/2024_stats/2024_data.csv')
ken_2024 = pd.read_csv('Data/2024_stats/2024_Kenpom.csv')
res_2024 = pd.read_csv('Data/2024_stats/2024_results.csv')
#%%
res_2023.head()
#%%
import random
def swap(df):
    for i in range(len(df)):
        int = random.choice([0, 1])
        if int == 0:
            df.loc[i, 'Team 1 Score'], df.loc[i, 'Team 2 Score'] = df.loc[i, 'Team 2 Score'], df.loc[i, 'Team 1 Score']
            df.loc[i, 'Team 1 Name'], df.loc[i, 'Team 2 Name'] = df.loc[i, 'Team 2 Name'], df.loc[i, 'Team 1 Name']
    
    return df
#%%
swap(res_2023)
#%%
swap(res_2024)
#%%
def make_games(res, ken):
    # Convert team stats data into a dictionary for quick lookup
    team_stats = ken.set_index('Team').to_dict(orient='index')
    
    # Create a new list to store the transformed data
    game_data = []
    
    # Iterate through each game in results_df
    for _, row in res.iterrows():
        team1 = row['Team 1 Name']
        team2 = row['Team 2 Name']
        team1_score = row['Team 1 Score']
        team2_score = row['Team 2 Score']
    
        if team1 in team_stats and team2 in team_stats:
            # Get stats for both teams
            team1_stats = team_stats[team1]
            team2_stats = team_stats[team2]
    
            # Compute the differences in stats (team1 - team2)
            stat_diffs = {stat: float(team1_stats[stat]) - float(team2_stats[stat]) 
                          for stat in team1_stats.keys() if isinstance(team1_stats[stat], (int, float))}
    
            # Append results
            game_data.append({
                'Team 1 Name': team1,
                'Team 1 Score': team1_score,
                'Team 2 Name': team2,
                'Team 2 Score': team2_score,
                **stat_diffs
            })
        else:
            print(team1 + " not found")
    
    # Convert the list into a DataFrame
    games_df = pd.DataFrame(game_data)
    
    return games_df
#%%
def add_score(df):
    df['Score'] = df['Team 1 Score'] - df['Team 2 Score']
    
    return df
#%%
games_2023 = make_games(res_2023, ken_2023)
games_2024 = make_games(res_2024, ken_2024)
#%%
add_score(games_2023)
add_score(games_2024)
#%%
games_2023.columns
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score

def run(X_train, y_train, X_test, y_test):
    
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
    model.fit(X_train_selected, y_train)
    
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    selected_features = X_train.columns[selector.get_support()]
    
    return accuracy, selected_features
    
#%%
X1 = games_2023.drop(columns=['Team 1 Name', 'Team 1 Score', 'Team 2 Name', 'Team 2 Score', 'Score'])
y1 = (games_2023['Score'] > 0).astype(int)

X2 = games_2024.drop(columns=['Team 1 Name', 'Team 1 Score', 'Team 2 Name', 'Team 2 Score', 'Score'])
y2 = (games_2024['Score'] > 0).astype(int)
#%%
run(X1, y1, X2, y2)
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


def ensemble_prediction(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Create base models
    xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=500)
    lr_model  = LogisticRegression(random_state=42)
    rf_model  = RandomForestClassifier(n_estimators=500, random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    gbm_model = GradientBoostingClassifier(n_estimators=500, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lr', lr_model),
            ('rf', rf_model),
            ('svm', svm_model),
            ('gbm', gbm_model),
            ('knn', knn_model),
            
        ],
        voting='soft'
    )
    
    # Train and evaluate
    fit = ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, fit
#%%
acc, fit = ensemble_prediction(X1, y1, X2, y2)
print(acc)
#%%
fit.predict(X2.iloc[[0]])[0]
#%%
correct = 0
for i in range(X2.shape[0]):
    pred = fit.predict(X2.iloc[[i]])[0]
    real = y2.iloc[i]
    if pred == real:
        correct += 1
        
print(correct/X2.shape[0])
#%%
