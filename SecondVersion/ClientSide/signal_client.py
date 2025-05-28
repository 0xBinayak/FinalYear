
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import os
import sys



SERVER_URL = "http://192.168.0.172:5000" 
CLIENT_ID = "default_client" 


N_FEATURES = 16         
NUM_MODULATION_TYPES = 3 
MOD_CLASSES = {0: 'BPSK', 1: 'QPSK', 2: '8PSK'}


NUM_LOCAL_EPOCHS = 10     
LEARNING_RATE = 0.001   
HIDDEN_LAYER_SIZE = 128  


DATA_DIR = "federated_data"
MODEL_DIR = "." 


class SignalClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, HIDDEN_LAYER_SIZE) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, NUM_MODULATION_TYPES)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



def load_client_training_data(client_id_str):
    if client_id_str.startswith("client_pc"):
        data_file_id_part = client_id_str.replace("client_pc", "")
    elif client_id_str.startswith("client_"): 
        data_file_id_part = client_id_str.replace("client_", "")
    else:
        print(f"[{CLIENT_ID}] Warning: Could not determine numeric part from client_id '{client_id_str}' for data loading. Defaulting to '1'.")
        data_file_id_part = "1"

    features_filename = f"client{data_file_id_part}_train_data_features.pt"
    labels_filename = f"client{data_file_id_part}_train_data_labels.pt"
    
    features_path = os.path.join(DATA_DIR, features_filename)
    labels_path = os.path.join(DATA_DIR, labels_filename)

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"[{CLIENT_ID}] Error: Training data files not found for ID part '{data_file_id_part}' in {DATA_DIR}")
        print(f"Expected: {features_path} and {labels_path}")
        print("Please run 'generate_federated_signal_dataset.py' first.")
        return None

    print(f"[{CLIENT_ID}] Loading training data from {features_path} and {labels_path}")
    features = torch.load(features_path)
    labels = torch.load(labels_path)
    
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


def download_global_model(save_path="global_signal_model.pth"):
    print(f"[{CLIENT_ID}] Attempting to download global model from {SERVER_URL}/get_global_weights/")
    try:
        response = requests.get(f"{SERVER_URL}/get_global_weights/", timeout=10) 
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"[{CLIENT_ID}] Global model downloaded successfully to {save_path}.")
            return True
        elif response.status_code == 404:
            print(f"[{CLIENT_ID}] No global model found on server (404). This is normal for the first client round.")
            return False
        else:
            print(f"[{CLIENT_ID}] Failed to download global model. Status: {response.status_code}, Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[{CLIENT_ID}] Error downloading global model: {e}")
        return False
    
    
    
    

def train_local_model(model, train_loader, epochs=NUM_LOCAL_EPOCHS, learning_rate=LEARNING_RATE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    print(f"[{CLIENT_ID}] Starting local training for {epochs} epochs with LR={learning_rate}...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for data_batch, target_batch in train_loader:
            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if num_batches > 0:
            print(f"[{CLIENT_ID}] Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss/num_batches:.4f}")
        else:
            print(f"[{CLIENT_ID}] Epoch {epoch+1}/{epochs}, No data in train_loader for training.")
            break 
    print(f"[{CLIENT_ID}] Local training finished.")

def upload_weights(weights_path):
    if not CLIENT_ID or CLIENT_ID == "default_client":
        print(f"[{CLIENT_ID}] Error: CLIENT_ID not set properly before uploading weights.")
        return False 

    print(f"[{CLIENT_ID}] Attempting to upload weights from {weights_path}...")
    try:
        with open(weights_path, "rb") as f_weights:
            files = {'file': (os.path.basename(weights_path), f_weights, 'application/octet-stream')}
            data_payload = {'client_id': CLIENT_ID}
            response = requests.post(f"{SERVER_URL}/upload_weights/", files=files, data=data_payload, timeout=15) 
            response.raise_for_status() 
            print(f"[{CLIENT_ID}] Upload successful. Server response: {response.json()}")
            return True
    except requests.exceptions.HTTPError as e:
        print(f"[{CLIENT_ID}] HTTP error uploading weights: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[{CLIENT_ID}] Network error uploading weights: {e}")
    except Exception as e:
        print(f"[{CLIENT_ID}] An unexpected error occurred during upload: {e}")
    return False




def main():
    global CLIENT_ID 
    if len(sys.argv) < 2:
        print("Usage: python signal_client.py <client_id_suffix>")
        print("Example: python signal_client.py pc1  (This will become client_pc1)")
        sys.exit(1)
    client_suffix = sys.argv[1]
    CLIENT_ID = f"client_{client_suffix}" 
    print(f"--- Client {CLIENT_ID} starting ---")
    print(f"--- Using Local Epochs: {NUM_LOCAL_EPOCHS}, Learning Rate: {LEARNING_RATE}, Hidden Layer Size: {HIDDEN_LAYER_SIZE} ---")


    model = SignalClassifierModel() 
    downloaded_global_model_path = os.path.join(MODEL_DIR, f"downloaded_global_model_{CLIENT_ID}.pth")
    purely_local_model_save_path = os.path.join(MODEL_DIR, f"{CLIENT_ID}_purely_local_model_H{HIDDEN_LAYER_SIZE}_E{NUM_LOCAL_EPOCHS}_LR{LEARNING_RATE}.pth")


    if download_global_model(downloaded_global_model_path):
        try:
            model.load_state_dict(torch.load(downloaded_global_model_path))
            print(f"[{CLIENT_ID}] Successfully loaded global model weights.")
        except RuntimeError as e:
            print(f"[{CLIENT_ID}] Error loading downloaded global model weights: {e}.")
            print(f"[{CLIENT_ID}] This might be due to a change in model architecture (N_FEATURES, HIDDEN_LAYER_SIZE, NUM_MODULATION_TYPES) or an old global model file on the server.")
            print(f"[{CLIENT_ID}] Delete 'global_signal_model.pth' on the server if architecture changed. Training from scratch with fresh model for this client.")
            model = SignalClassifierModel() 
        except Exception as e:
            print(f"[{CLIENT_ID}] Other error loading downloaded global model: {e}. Training from scratch with fresh model.")
            model = SignalClassifierModel() 
    else:
        print(f"[{CLIENT_ID}] No global model available or error in download. Training from scratch with fresh model.")

    train_loader = load_client_training_data(CLIENT_ID) 
    
    if train_loader is None or len(train_loader.dataset) == 0:
        print(f"[{CLIENT_ID}] No training data loaded. Exiting.")
        return

    # Perform local training using configured epochs and LR
    train_local_model(model, train_loader, epochs=NUM_LOCAL_EPOCHS, learning_rate=LEARNING_RATE)

    """" Saving PURELY LOCAL model  to compare pre-FL performance with post-FL performance."""
    if not os.path.exists(purely_local_model_save_path): 
        torch.save(model.state_dict(), purely_local_model_save_path)
        print(f"[{CLIENT_ID}] Saved PURELY LOCAL model to {purely_local_model_save_path}")
    


    local_weights_for_upload_path = os.path.join(MODEL_DIR, f"{CLIENT_ID}_local_signal_weights_for_upload.pth")
    torch.save(model.state_dict(), local_weights_for_upload_path)
    
    print(f"[{CLIENT_ID}] Uploading weights from {local_weights_for_upload_path}...")
    upload_success = upload_weights(local_weights_for_upload_path)
    
    if not upload_success:
        print(f"[{CLIENT_ID}] Failed to upload weights. Local weights for upload saved at {local_weights_for_upload_path}")
    else:
        if os.path.exists(local_weights_for_upload_path):
            try:
                os.remove(local_weights_for_upload_path)
                print(f"[{CLIENT_ID}] Cleaned up {local_weights_for_upload_path}.")
            except OSError as e:
                print(f"[{CLIENT_ID}] Error deleting {local_weights_for_upload_path}: {e}")

    if os.path.exists(downloaded_global_model_path): 
        try:
            os.remove(downloaded_global_model_path)
        except OSError as e:
            print(f"[{CLIENT_ID}] Error deleting {downloaded_global_model_path}: {e}")

    print(f"--- Client {CLIENT_ID} finished one round ---")

if __name__ == "__main__":
    if "<RASPBERRY_PI_IP>" in SERVER_URL:
        print("ERROR: Please replace <RASPBERRY_PI_IP> in the SERVER_URL variable with your Raspberry Pi's actual IP address.")
    else:
        main()