import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def generate_synthetic_data_with_noise(phase="initial"):
    NUM_VOTERS = 1000
    NUM_PROPOSALS = 100
    voters = pd.DataFrame({
        "voter_id": [f"voter_{i}" for i in range(1, NUM_VOTERS + 1)],
        "token_holdings": np.random.exponential(scale=100, size=NUM_VOTERS) + np.random.normal(0, 10, NUM_VOTERS),
    })
    proposals = pd.DataFrame({
        "proposal_id": [f"proposal_{i}" for i in range(1, NUM_PROPOSALS + 1)],
        "proposal_creation_time": pd.date_range(start="2023-01-01", periods=NUM_PROPOSALS, freq="D"),
    })

    num_votes = int(NUM_VOTERS * 1.5)
    voting_data = pd.DataFrame({
        "voter_id": np.random.choice(voters["voter_id"], size=num_votes),
        "proposal_id": np.random.choice(proposals["proposal_id"], size=num_votes),
        "voting_weight": np.random.normal(loc=50, scale=15, size=num_votes),
        "vote_time": pd.date_range(start="2023-01-01", periods=num_votes, freq="T"),
        "vote_outcome": np.random.choice(["Yes", "No", "Abstain"], size=num_votes, p=[0.6, 0.3, 0.1]),
        "label": [0] * num_votes
    })
    anomaly_scale = 75 if phase == "initial" else 50
    anomalies = pd.DataFrame({
        "voter_id": [f"voter_{i}" for i in range(NUM_VOTERS + 1, NUM_VOTERS + 51)],
        "proposal_id": np.random.choice(proposals["proposal_id"], size=50),
        "voting_weight": np.random.normal(loc=200, scale=anomaly_scale, size=50),
        "vote_time": pd.date_range(start="2023-01-01", periods=50, freq="H"),
        "vote_outcome": np.random.choice(["Yes", "No"], size=50),
        "label": [1] * 50  # Anomalous behavior
    })

    voting_data = pd.concat([voting_data, anomalies], ignore_index=True)
    return voters, proposals, voting_data

def prepare_features(voting_data, voters, proposals):
    data = voting_data.merge(voters, on="voter_id", how="left")
    data = data.merge(proposals, on="proposal_id", how="left")
    data["time_since_proposal"] = (
        pd.to_datetime(data["vote_time"]) - pd.to_datetime(data["proposal_creation_time"])
    ).dt.total_seconds() / 86400
    data["vote_outcome_numeric"] = data["vote_outcome"].map({"Yes": 1, "No": 0, "Abstain": 0.5})
    features = data[["voting_weight", "token_holdings", "time_since_proposal", "vote_outcome_numeric"]]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    features["voting_weight"] = np.clip(features["voting_weight"], 0, 300)
    features["token_holdings"] = np.clip(features["token_holdings"], 0, 500)
    features["time_since_proposal"] = np.clip(features["time_since_proposal"], 0, 365)

    labels = data["label"]
    return features, labels

def build_neural_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    voters, proposals, voting_data = generate_synthetic_data_with_noise(phase="initial")
    features, labels = prepare_features(voting_data, voters, proposals)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(features_scaled), 1):
        print(f"\n=== Fold {fold} ===")
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        model = build_neural_network(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weight_dict, verbose=1)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
