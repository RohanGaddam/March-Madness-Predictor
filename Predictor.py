import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re


def load_data(stats_path, results_path):
    """Load and prepare the data with team name matching"""
    # Load data
    stats_df = pd.read_csv(stats_path)
    results_df = pd.read_csv(results_path)

    # Extract base team names from stats_df by removing conference info in parentheses
    stats_df['BaseTeam'] = stats_df['Team'].apply(lambda x: re.sub(r'\s*\([^)]*\)\s*', '', x).strip())

    # Convert string columns to numeric types
    numeric_columns = [col for col in stats_df.columns if col != 'Team' and col != 'BaseTeam']
    for col in numeric_columns:
        stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')

    # Fill NaN values with the mean of each column
    stats_df[numeric_columns] = stats_df[numeric_columns].fillna(stats_df[numeric_columns].mean())

    # Clean up team names in results_df
    results_df['Team 1 Name'] = results_df['Team 1 Name'].str.strip()
    results_df['Team 2 Name'] = results_df['Team 2 Name'].str.strip()

    # Create team name mapping for easier lookup
    team_mapping = {}
    for _, row in stats_df.iterrows():
        team_mapping[row['BaseTeam'].lower()] = row['Team']

    # Add BaseTeam columns to results_df for matching
    results_df['Team 1 BaseTeam'] = results_df['Team 1 Name']
    results_df['Team 2 BaseTeam'] = results_df['Team 2 Name']

    return stats_df, results_df, team_mapping


def create_matchup_features(stats_df, results_df, team_mapping):
    """Create features for each matchup with improved team name matching"""
    features = []
    labels = []
    matchups = []  # To keep track of which teams were matched

    # Drop the non-numeric columns to get only numeric features
    feature_columns = [col for col in stats_df.columns if col not in ['Team', 'BaseTeam']]

    # Ensure all feature columns are numeric
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(stats_df[col]):
            print(f"Warning: Column '{col}' is not numeric. Converting to numeric.")
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')

    # Fill any remaining NaN values
    stats_df[feature_columns] = stats_df[feature_columns].fillna(stats_df[feature_columns].mean())

    for _, row in results_df.iterrows():
        team1 = row['Team 1 BaseTeam']
        team2 = row['Team 2 BaseTeam']

        # Try direct match first
        team1_stats = stats_df[stats_df['BaseTeam'] == team1]
        team2_stats = stats_df[stats_df['BaseTeam'] == team2]

        # If no direct match, try case-insensitive match
        if len(team1_stats) == 0:
            team1_lower = team1.lower()
            if team1_lower in team_mapping:
                team1_full = team_mapping[team1_lower]
                team1_stats = stats_df[stats_df['Team'] == team1_full]

        if len(team2_stats) == 0:
            team2_lower = team2.lower()
            if team2_lower in team_mapping:
                team2_full = team_mapping[team2_lower]
                team2_stats = stats_df[stats_df['Team'] == team2_full]

        # Check for common abbreviations or alternative names
        if len(team1_stats) == 0 or len(team2_stats) == 0:
            # Handle special cases like "Miami (FL)" vs "Miami FL"
            name_fixes = {
                "Miami (FL)": ["Miami FL", "Miami"],
                "UConn": ["Connecticut"],
                "USC": ["Southern California"],
                "UCSB": ["UC Santa Barbara"],
                "St. Mary's": ["Saint Mary's"],
                "FDU": ["Fairleigh Dickinson"],
                # Add more mappings as needed
            }

            for full_name, alternatives in name_fixes.items():
                if team1 in alternatives:
                    team1_stats = stats_df[stats_df['BaseTeam'].str.contains(full_name, case=False)]
                if team2 in alternatives:
                    team2_stats = stats_df[stats_df['BaseTeam'].str.contains(full_name, case=False)]

        # Try partial matching as a last resort
        if len(team1_stats) == 0:
            for idx, row in stats_df.iterrows():
                if team1.replace(" ", "") in row['BaseTeam'].replace(" ", "") or row['BaseTeam'].replace(" ",
                                                                                                         "") in team1.replace(
                        " ", ""):
                    team1_stats = stats_df.iloc[idx:idx + 1]
                    break

        if len(team2_stats) == 0:
            for idx, row in stats_df.iterrows():
                if team2.replace(" ", "") in row['BaseTeam'].replace(" ", "") or row['BaseTeam'].replace(" ",
                                                                                                         "") in team2.replace(
                        " ", ""):
                    team2_stats = stats_df.iloc[idx:idx + 1]
                    break

        # If we found stats for both teams, create features
        if len(team1_stats) > 0 and len(team2_stats) > 0:
            # Ensure we're working with numeric values
            team1_stat_values = team1_stats[feature_columns].values[0].astype(float)
            team2_stat_values = team2_stats[feature_columns].values[0].astype(float)

            # Create features: difference and ratio between team stats
            diff_features = team1_stat_values - team2_stat_values
            ratio_features = np.divide(team1_stat_values, team2_stat_values, out=np.zeros_like(team1_stat_values),
                                       where=team2_stat_values != 0)

            # Combine features
            matchup_features = np.concatenate([diff_features, ratio_features])

            features.append(matchup_features)
            # Team 1 is always the winner in your dataset
            labels.append(1)

            # Add the reverse matchup with label 0
            inverse_ratio = np.divide(team2_stat_values, team1_stat_values, out=np.zeros_like(team2_stat_values),
                                      where=team1_stat_values != 0)
            features.append(np.concatenate([-diff_features, inverse_ratio]))
            labels.append(0)

            # Keep track of successful matchups
            matchups.append((team1, team2))
        else:
            if len(team1_stats) == 0 and len(team2_stats) == 0:
                print(f"Warning: Could not find stats for both {team1} and {team2}")
            elif len(team1_stats) == 0:
                print(f"Warning: Could not find stats for {team1}")
            else:
                print(f"Warning: Could not find stats for {team2}")

    print(f"Successfully created features for {len(matchups)} matchups out of {len(results_df)} results")

    if len(features) == 0:
        raise ValueError("No matchups could be created. Check team names in your datasets.")

    return np.array(features), np.array(labels), matchups


def train_model(X, y):
    """Train an XGBoost model"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(classification_report(y_val, val_preds))

    return model, scaler


def predict_winner(team1, team2, model, scaler, stats_df, team_mapping):
    """Predict the winner between two teams with improved team matching"""
    feature_columns = [col for col in stats_df.columns if col not in ['Team', 'BaseTeam']]

    # Try to find team1 stats
    team1_stats = stats_df[stats_df['BaseTeam'] == team1]
    if len(team1_stats) == 0:
        team1_lower = team1.lower()
        if team1_lower in team_mapping:
            team1_full = team_mapping[team1_lower]
            team1_stats = stats_df[stats_df['Team'] == team1_full]

    # Try to find team2 stats
    team2_stats = stats_df[stats_df['BaseTeam'] == team2]
    if len(team2_stats) == 0:
        team2_lower = team2.lower()
        if team2_lower in team_mapping:
            team2_full = team_mapping[team2_lower]
            team2_stats = stats_df[stats_df['Team'] == team2_full]

    # If we found stats for both teams, make a prediction
    if len(team1_stats) > 0 and len(team2_stats) > 0:
        team1_stat_values = team1_stats[feature_columns].values[0].astype(float)
        team2_stat_values = team2_stats[feature_columns].values[0].astype(float)

        # Create features
        diff_features = team1_stat_values - team2_stat_values
        ratio_features = np.divide(team1_stat_values, team2_stat_values, out=np.zeros_like(team1_stat_values),
                                   where=team2_stat_values != 0)
        matchup_features = np.concatenate([diff_features, ratio_features])

        # Scale and predict
        X = scaler.transform([matchup_features])
        prob = model.predict_proba(X)[0]

        if prob[1] > 0.5:
            return team1, prob[1]
        else:
            return team2, 1 - prob[1]
    else:
        if len(team1_stats) == 0 and len(team2_stats) == 0:
            print(f"Error: Could not find stats for both {team1} and {team2}")
        elif len(team1_stats) == 0:
            print(f"Error: Could not find stats for {team1}")
        else:
            print(f"Error: Could not find stats for {team2}")
        return None, 0


def evaluate_model(model, scaler, stats_df, results_df, team_mapping):
    """Evaluate model on test results"""
    correct = 0
    total = 0

    for _, row in results_df.iterrows():
        team1 = row['Team 1 BaseTeam']
        team2 = row['Team 2 BaseTeam']

        predicted_winner, confidence = predict_winner(team1, team2, model, scaler, stats_df, team_mapping)

        if predicted_winner == team1:  # Team 1 is always the actual winner in your data
            correct += 1
            result = "CORRECT"
        else:
            result = "WRONG"

        total += 1

        print(
            f"{team1} vs {team2}: Predicted {predicted_winner} (confidence: {confidence:.4f}), Actual: {team1} - {result}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy


def analyze_feature_importance(model, stats_df):
    """Analyze and print feature importance"""
    feature_columns = [col for col in stats_df.columns if col not in ['Team', 'BaseTeam']]

    # Get feature importance
    importance = model.feature_importances_

    # Create feature names for differences and ratios
    diff_features = [f"Diff_{col}" for col in feature_columns]
    ratio_features = [f"Ratio_{col}" for col in feature_columns]
    all_features = diff_features + ratio_features

    # Sort features by importance
    indices = np.argsort(importance)[::-1]

    print("\nFeature Importance:")
    for i, idx in enumerate(indices[:20]):  # Print top 20 features
        if idx < len(all_features):
            print(f"{i + 1}. {all_features[idx]}: {importance[idx]:.4f}")

    return all_features, importance


def debug_data_types(stats_df):
    """Print data types of all columns to help debug"""
    print("\nData Types in stats_df:")
    for col in stats_df.columns:
        print(f"{col}: {stats_df[col].dtype}")
        # Print first value of each column
        if len(stats_df) > 0:
            print(f"  First value: {stats_df[col].iloc[0]}")

    # Check for any non-numeric columns
    non_numeric = []
    for col in stats_df.columns:
        if col not in ['Team', 'BaseTeam'] and not pd.api.types.is_numeric_dtype(stats_df[col]):
            non_numeric.append(col)

    if non_numeric:
        print(f"\nWARNING: These columns are not numeric: {non_numeric}")


def main():
    # Load 2023 data for training
    train_stats_df, train_results_df, train_team_mapping = load_data(
        "Data/2023_stats/2023_data.csv",
        "Data/2023_stats/2023_Results.csv"
    )

    # Print some stats to verify data loading
    print(f"Loaded {len(train_stats_df)} teams in stats data")
    print(f"Loaded {len(train_results_df)} matchup results")

    # Debug data types
    debug_data_types(train_stats_df)

    # Create features and train model
    try:
        X, y, matchups = create_matchup_features(train_stats_df, train_results_df, train_team_mapping)
        print(f"Created features array with shape: {X.shape}")

        model, scaler = train_model(X, y)

        # Analyze feature importance
        analyze_feature_importance(model, train_stats_df)

        # Load 2024 data for testing
        test_stats_df, test_results_df, test_team_mapping = load_data(
            "Data/2024_stats/2024_data.csv",
            "Data/2024_stats/2024_Results.csv"
        )

        # Evaluate on 2024 data
        evaluate_model(model, scaler, test_stats_df, test_results_df, test_team_mapping)

        # Interactive prediction
        while True:
            print("\nEnter two team names to predict a winner (or 'quit' to exit):")
            team1 = input("Team 1: ")
            if team1.lower() == 'quit':
                break

            team2 = input("Team 2: ")
            if team2.lower() == 'quit':
                break

            winner, confidence = predict_winner(team1, team2, model, scaler, test_stats_df, test_team_mapping)
            if winner:
                print(f"Predicted winner: {winner} with {confidence:.2%} confidence")

    except ValueError as e:
        print(f"Error: {e}")
        print("Try examining your data files to ensure team names can be matched between stats and results.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()