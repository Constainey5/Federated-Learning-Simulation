
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuration ---
NUM_CLIENTS = 5
NUM_ROUNDS = 10
DATA_PER_CLIENT = 200
FEATURES = 20
RANDOM_STATE = 42

print("
--- Federated Learning Simulation ---")

# --- Simulate Client Data ---
def generate_client_data(num_clients, data_per_client, features, random_state):
    print(f"Generating synthetic data for {num_clients} clients...")
    client_datasets = []
    for i in range(num_clients):
        X, y = make_classification(n_samples=data_per_client,
                                   n_features=features,
                                   n_informative=features // 2,
                                   n_redundant=features // 4,
                                   random_state=random_state + i)
        client_datasets.append((X, y))
    print("Client data generated.")
    return client_datasets

# --- Client-side Training ---
class Client:
    def __init__(self, client_id, X_train, y_train):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=RANDOM_STATE)
        print(f"Client {self.client_id}: Initialized with {len(self.X_train)} samples.")

    def train_local_model(self, global_model_weights=None):
        if global_model_weights is not None:
            self.model.coef_ = global_model_weights[0]
            self.model.intercept_ = global_model_weights[1]
        
        # Train for one epoch on local data
        self.model.fit(self.X_train, self.y_train)
        print(f"Client {self.client_id}: Trained locally.")
        return self.model.coef_, self.model.intercept_, len(self.X_train)

# --- Server-side Aggregation ---
class Server:
    def __init__(self, num_features):
        self.global_model_weights = (np.zeros((1, num_features)), np.zeros(1))
        print("Server: Initialized global model weights.")

    def aggregate_models(self, client_updates):
        # Federated Averaging (FedAvg)
        total_samples = sum(update[2] for update in client_updates)
        
        new_coef = np.zeros_like(self.global_model_weights[0])
        new_intercept = np.zeros_like(self.global_model_weights[1])

        for coef, intercept, num_samples in client_updates:
            new_coef += coef * (num_samples / total_samples)
            new_intercept += intercept * (num_samples / total_samples)
        
        self.global_model_weights = (new_coef, new_intercept)
        print("Server: Aggregated client models.")
        return self.global_model_weights

    def evaluate_global_model(self, X_test, y_test):
        temp_model = SGDClassifier(loss='log_loss', random_state=RANDOM_STATE)
        temp_model.coef_ = self.global_model_weights[0]
        temp_model.intercept_ = self.global_model_weights[1]
        
        # To use predict, the model needs to be partially fitted or have classes set
        # For evaluation, we can manually calculate accuracy if classes are known
        # Or, we can fit a dummy model to set classes and then set weights
        temp_model.partial_fit(X_test[:1], y_test[:1], classes=np.unique(y_test))
        temp_model.coef_ = self.global_model_weights[0]
        temp_model.intercept_ = self.global_model_weights[1]

        predictions = temp_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Server: Global model accuracy on test set: {accuracy:.4f}")
        return accuracy


if __name__ == "__main__":
    # Generate data for clients and a global test set
    all_X, all_y = make_classification(n_samples=NUM_CLIENTS * DATA_PER_CLIENT + 500, 
                                       n_features=FEATURES, random_state=RANDOM_STATE)
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(all_X, all_y, test_size=0.2, random_state=RANDOM_STATE)

    client_datasets = []
    start_idx = 0
    for i in range(NUM_CLIENTS):
        end_idx = start_idx + DATA_PER_CLIENT
        client_datasets.append((X_train_global[start_idx:end_idx], y_train_global[start_idx:end_idx]))
        start_idx = end_idx

    clients = [Client(i, data[0], data[1]) for i, data in enumerate(client_datasets)]
    server = Server(FEATURES)

    accuracies = []
    for round_num in range(NUM_ROUNDS):
        print(f"
--- Federated Round {round_num + 1}/{NUM_ROUNDS} ---")
        client_updates = []
        for client in clients:
            coef, intercept, num_samples = client.train_local_model(server.global_model_weights)
            client_updates.append((coef, intercept, num_samples))
        
        server.aggregate_models(client_updates)
        acc = server.evaluate_global_model(X_test_global, y_test_global)
        accuracies.append(acc)

    print("
Federated Learning simulation completed.")
    print(f"Final global model accuracy: {accuracies[-1]:.4f}")
