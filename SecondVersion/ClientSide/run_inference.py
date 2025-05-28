import torch
import torch.nn as nn
import requests
import os



SERVER_URL = "http://192.168.0.172:5000"  
GLOBAL_MODEL_DOWNLOAD_NAME = "global_signal_model.pth"
DATA_DIR = "federated_data"
PERSISTENT_GLOBAL_MODEL_SAVE_PATH = "final_global_model_used_for_inference.pth"

N_FEATURES = 16        
NUM_MODULATION_TYPES = 3 
MOD_CLASSES = {0: 'BPSK', 1: 'QPSK', 2: '8PSK'} 

""" Hyperparameter """
HIDDEN_LAYER_SIZE = 128  

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


def load_inference_data():
    features_path = os.path.join(DATA_DIR, "inference_data_batch_features.pt")
    labels_path = os.path.join(DATA_DIR, "inference_data_batch_labels.pt")

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Error: Inference data files not found in {DATA_DIR}")
        print(f"Expected: {features_path} and {labels_path}")
        print("Please run 'generate_federated_signal_dataset.py' (from Canvas artifact generate_dataset_for_fl_v2) first.")
        return None, None
    
    print(f"Loading inference data from {features_path} and {labels_path}")
    features = torch.load(features_path)
    labels = torch.load(labels_path)
    return features, labels


def download_global_model(save_path): 
    print(f"Attempting to download global model from {SERVER_URL}/get_global_weights/ to {save_path}")
    try:
        response = requests.get(f"{SERVER_URL}/get_global_weights/", timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Global model downloaded successfully to {save_path}")
            return True
        elif response.status_code == 404:
            print(f"No global model found on server (404). Cannot perform inference.")
            return False
        else:
            print(f"Failed to download global model. Status: {response.status_code}, Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading global model: {e}")
        return False


def perform_batch_inference(model, signal_features_batch_tensor):
    model.eval() 
    with torch.no_grad(): 
        outputs = model(signal_features_batch_tensor)
        _, predicted_indices = torch.max(outputs.data, 1)
        return predicted_indices.cpu().numpy()


def main():
    print("--- Inference Client for Federated Signal Classifier (from pre-generated file) ---")
    print(f"--- Using Model Hidden Layer Size: {HIDDEN_LAYER_SIZE} ---")
    temp_download_model_path = "temp_downloaded_global_for_inference.pth" 
    if not download_global_model(temp_download_model_path):
        print("Exiting due to failure in downloading the global model.")
        return
    model = SignalClassifierModel() 
    try:
        model.load_state_dict(torch.load(temp_download_model_path))
        print("Successfully loaded trained global model from temporary download.")

        
        torch.save(model.state_dict(), PERSISTENT_GLOBAL_MODEL_SAVE_PATH)
        print(f"Global model used for inference has been saved to: {PERSISTENT_GLOBAL_MODEL_SAVE_PATH}")

    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print(f"This often means a mismatch in model architecture (N_FEATURES, HIDDEN_LAYER_SIZE, NUM_MODULATION_TYPES) ")
        print(f"between the saved global model and the model defined in this script.")
        print(f"Ensure HIDDEN_LAYER_SIZE ({HIDDEN_LAYER_SIZE}) in this script matches the one used for training.")
        if os.path.exists(temp_download_model_path): os.remove(temp_download_model_path)
        return
    except Exception as e:
        print(f"Generic error loading model state_dict: {e}")
        if os.path.exists(temp_download_model_path): os.remove(temp_download_model_path) 
        return
        
    
    inference_features, inference_labels_actual_indices = load_inference_data()
    if inference_features is None or inference_labels_actual_indices is None:
        print("Exiting due to missing inference data.")
        if os.path.exists(temp_download_model_path): os.remove(temp_download_model_path) 
        return
    
    print(f"\nLoaded {len(inference_labels_actual_indices)} samples for inference.")

    
    print("\n--- Performing Inference on Loaded Batch Data ---")
    model.eval() 
    
    all_predicted_indices = perform_batch_inference(model, inference_features)

    num_correct = 0
    for i in range(0, len(inference_features), 2):
        print(f"\nInference Set {i//2 + 1}:")
        for j in range(2): 
            current_index = i + j
            if current_index >= len(inference_features):
                break 

            actual_label_idx = inference_labels_actual_indices[current_index].item()
            pred_label_idx = all_predicted_indices[current_index]
            
            actual_name = MOD_CLASSES.get(actual_label_idx, "Unknown")
            pred_name = MOD_CLASSES.get(pred_label_idx, "Unknown")
            
            correct_str = "CORRECT" if actual_label_idx == pred_label_idx else "INCORRECT"
            if actual_label_idx == pred_label_idx:
                num_correct +=1

            print(f"  Sample {current_index + 1}: Actual='{actual_name}' (Idx:{actual_label_idx}), Predicted='{pred_name}' (Idx:{pred_label_idx}) -> {correct_str}")

    if len(inference_features) > 0:
        inference_accuracy = num_correct / len(inference_features)
        print(f"\nAccuracy on this inference batch: {num_correct}/{len(inference_features)} = {inference_accuracy:.4f}")
    else:
        print("No samples in the inference batch to evaluate.")

    
    if os.path.exists(temp_download_model_path):
        try:
            os.remove(temp_download_model_path)
            print(f"Cleaned up temporary download: {temp_download_model_path}")
        except OSError as e:
            print(f"Error deleting temporary download {temp_download_model_path}: {e}")
    print("\n--- Inference complete ---")

if __name__ == "__main__":
    if "<RASPBERRY_PI_IP>" in SERVER_URL:
        print("ERROR: Please replace <RASPBERRY_PI_IP> in the SERVER_URL variable with your Raspberry Pi's actual IP address.")
    else:
        main()
