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
def simulate_tournament(initial_matchups, kendf, model):
    """
    Simulates a March Madness tournament based on initial matchups.
    
    Parameters:
    initial_matchups (DataFrame): DataFrame with columns 'Team 1 Name' and 'Team 2 Name'
                                 representing the first round matchups
    kendf (DataFrame): KenPom data for each team
    model: Trained prediction model
    
    Returns:
    dict: Tournament results with winners for each round
    """
    import pandas as pd
    import copy
    
    # Validate input
    if 'Team 1 Name' not in initial_matchups.columns or 'Team 2 Name' not in initial_matchups.columns:
        raise ValueError("Initial matchups DataFrame must have 'Team 1 Name' and 'Team 2 Name' columns")
    
    num_teams = len(initial_matchups) * 2
    
    # Determine tournament structure based on number of teams
    if num_teams == 64:
        rounds = ['Round 1', 'Round 2', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
    elif num_teams == 32:
        rounds = ['Round 1', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
    elif num_teams == 16:
        rounds = ['Round 1', 'Elite 8', 'Final 4', 'Championship']
    elif num_teams == 8:
        rounds = ['Round 1', 'Final 4', 'Championship']
    elif num_teams == 4:
        rounds = ['Round 1', 'Championship']
    else:
        raise ValueError(f"Unsupported number of teams: {num_teams}")
    
    # Dictionary to store results of each round
    results = {round_name: None for round_name in rounds}
    results['Champion'] = None
    
    # Start with initial matchups
    current_bracket = initial_matchups.copy()
    results[rounds[0]] = current_bracket.copy()
    
    # Simulate each round
    for i, round_name in enumerate(rounds):
        print(f"Simulating {round_name}...")
        next_round_teams = []
        
        # For each matchup in current round
        for idx, row in current_bracket.iterrows():
            # Get the teams
            team1 = row['Team 1 Name']
            team2 = row['Team 2 Name']
            
            # Create a single game DataFrame for prediction
            game = pd.DataFrame({
                'Team 1 Name': [team1],
                'Team 2 Name': [team2],
                'Team 1 Score': [0],  # Placeholder
                'Team 2 Score': [0]   # Placeholder
            })
            
            # Process the game using make_games and prepare for prediction
            processed_game = make_games(game, kendf)
            
            # Check if teams were found in KenPom data
            if processed_game.empty:
                print(f"Warning: Could not find KenPom data for matchup: {team1} vs {team2}")
                # Choose a random winner if data is missing
                import random
                winner = team1 if random.random() > 0.5 else team2
            else:
                processed_game = add_score(processed_game)
                X_game = processed_game.drop(columns=['Team 1 Name', 'Team 1 Score', 'Team 2 Name', 'Team 2 Score', 'Score'])
                
                # Predict the winner
                prediction = model.predict(X_game)[0]
                
                # Determine the winner based on prediction
                winner = team1 if prediction == 1 else team2
            
            next_round_teams.append(winner)
        
        # If this is the final round, store the champion and exit
        if i == len(rounds) - 1:
            results['Champion'] = next_round_teams[0]
            break
            
        # Create bracket for the next round
        next_round_bracket = pd.DataFrame({
            'Team 1 Name': next_round_teams[0::2],
            'Team 2 Name': next_round_teams[1::2]
        })
        
        # Update current bracket and store results
        current_bracket = next_round_bracket
        results[rounds[i+1]] = current_bracket.copy()
    
    return results

#%%
def print_tournament_results(results):
    """
    Prints the results of the tournament in a readable format.
    
    Parameters:
    results (dict): Tournament results from simulate_tournament
    """
    print("\n🏀 MARCH MADNESS TOURNAMENT SIMULATION RESULTS 🏀\n")
    
    # Print each round's matchups and winners
    for round_name, bracket in results.items():
        if round_name == 'Champion':
            print(f"\n🏆 CHAMPION: {results['Champion']} 🏆")
        elif bracket is not None:
            print(f"\n== {round_name} ==")
            for i, row in bracket.iterrows():
                print(f"Game {i+1}: {row['Team 1 Name']} vs {row['Team 2 Name']}")
#%%
def get_bracket_visual(results):
    """
    Creates a simple text-based visualization of the tournament bracket.
    
    Parameters:
    results (dict): Tournament results from simulate_tournament
    
    Returns:
    str: Text representation of the bracket
    """
    output = []
    rounds = [r for r in results.keys() if r != 'Champion']
    
    # Get the champion
    champion = results['Champion']
    
    # Generate bracket visualization
    output.append("TOURNAMENT BRACKET")
    output.append("=================")
    
    for i, round_name in enumerate(rounds):
        output.append(f"\n{round_name}:")
        bracket = results[round_name]
        
        if bracket is None:
            continue
            
        for j, row in bracket.iterrows():
            team1 = row['Team 1 Name']
            team2 = row['Team 2 Name']
            
            # Check if we know the winner of this matchup
            winner = None
            if i < len(rounds) - 1:
                next_round = results[rounds[i+1]]
                if next_round is not None:
                    # Check if either team appears in the next round
                    if any(next_round['Team 1 Name'] == team1) or any(next_round['Team 2 Name'] == team1):
                        winner = team1
                    elif any(next_round['Team 1 Name'] == team2) or any(next_round['Team 2 Name'] == team2):
                        winner = team2
            elif i == len(rounds) - 1:
                # Championship game
                winner = champion
                
            # Format the matchup line
            if winner:
                if winner == team1:
                    output.append(f"  {team1} ✓ vs {team2}")
                else:
                    output.append(f"  {team1} vs {team2} ✓")
            else:
                output.append(f"  {team1} vs {team2}")
    
    output.append(f"\nChampion: 🏆 {champion} 🏆")
    
    return "\n".join(output)
#%%
teams = games_2024[['Team 1 Name', 'Team 2 Name']].iloc[4:36]
#%%
teams
#%%
def run_simulation(initial_matchups_df, kenpom_df, model):
    """
    Run a tournament simulation with the provided initial matchups.
    
    Parameters:
    initial_matchups_df (DataFrame): DataFrame with Team 1 Name and Team 2 Name columns
    kenpom_df (DataFrame): KenPom data for all teams
    model: Trained prediction model
    
    Returns:
    dict: Tournament results
    """
    # Run the simulation
    results = simulate_tournament(initial_matchups_df, kenpom_df, model)
    
    # Print the results
    print_tournament_results(results)
    
    # Get and print the bracket visualization
    bracket_visual = get_bracket_visual(results)
    print("\n")
    print(bracket_visual)
    
    return results
#%%
run_simulation(teams, ken_2024, fit)
#%%
def make_games_scoreless(res, ken):
    # Convert team stats data into a dictionary for quick lookup
    team_stats = ken.set_index('Team').to_dict(orient='index')
    
    # Create a new list to store the transformed data
    game_data = []
    
    # Iterate through each game in results_df
    for _, row in res.iterrows():
        team1 = row['Team 1 Name']
        team2 = row['Team 2 Name']
    
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
                'Team 2 Name': team2,
                **stat_diffs
            })
        else:
            print(team1 + " not found")
    
    # Convert the list into a DataFrame
    games_df = pd.DataFrame(game_data)
    
    return games_df
#%%
matchups_2025 = pd.read_csv('Data/2025_stats/2025_teams.csv')
ken_2025 = pd.read_csv('Data/2025_stats/2025_kenpom.csv')
#%%
make_games_scoreless(matchups_2025, ken_2025)
#%% md
# 
#%%
new_x = pd.concat([games_2023, games_2024])
#%%
X1 = new_x.drop(columns=['Team 1 Name', 'Team 1 Score', 'Team 2 Name', 'Team 2 Score', 'Score'])
y1 = (new_x['Score'] > 0).astype(int)
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


def new_predict(X_train, y_train):
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
    
    return fit
#%%
fit_2025 = new_predict(X1, y1)
#%%
teams
#%%
matchups_2025
#%%
run_simulation(matchups_2025, ken_2025, fit_2025)
#%%
