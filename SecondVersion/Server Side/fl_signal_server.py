from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import torch
import torch.nn as nn 
import os
import uvicorn

app = FastAPI()

GLOBAL_MODEL_PATH = "global_signal_model.pth"
NUM_CLIENTS = 3 

client_weights = {}


N_FEATURES = 16
NUM_MODULATION_TYPES = 3
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


def load_global_model():
    model = SignalClassifierModel()
    if os.path.exists(GLOBAL_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
            print(f"Loaded existing global model from {GLOBAL_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading global model: {e}. Initializing and saving a new one.")
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH) 
    else:
        torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
        print(f"No global model found at {GLOBAL_MODEL_PATH}. Initialized and saved a new one.")
    return model

def save_global_model(model):
    torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
    print(f"Global model saved to {GLOBAL_MODEL_PATH}")

def aggregate_weights(weights_list):
    avg_weights = {}
    if not weights_list:
        print("Warning: Weights list for aggregation is empty.")
        return None

    processed_weights_list = []
    for item in weights_list:
        if isinstance(item, dict):
            processed_weights_list.append(item)
        elif hasattr(item, 'state_dict') and callable(item.state_dict):
            processed_weights_list.append(item.state_dict())
        else:
            print(f"Warning: Item in weights_list is invalid type: {type(item)}")
            continue
            
    if not processed_weights_list:
        print("Warning: No valid state_dicts found in weights_list for aggregation.")
        return None

    first_keys = set(processed_weights_list[0].keys())
    for i, state_dict in enumerate(processed_weights_list[1:], 1):
        if set(state_dict.keys()) != first_keys:
            print(f"Error: Mismatch in model keys between client 0 and client {i}. Cannot aggregate.")
            print(f"Client 0 keys: {first_keys}")
            print(f"Client {i} keys: {set(state_dict.keys())}")
            return None

    print("Aggregating weights...")
    for key in processed_weights_list[0].keys():
        try:
            tensors_to_stack = [weights[key].float() for weights in processed_weights_list]
            avg_weights[key] = torch.stack(tensors_to_stack, dim=0).mean(dim=0)
        except Exception as e:
            print(f"Error aggregating key {key}: {e}")
            
            if processed_weights_list[0].get(key) is not None:
                avg_weights[key] = processed_weights_list[0][key].clone().detach()
                print(f"Fallback: Using weights from the first client for key {key}.")
            else:
                print(f"Critical error: Key {key} not found in first client's weights during fallback.")
                return None
    print("Weight aggregation complete.")
    return avg_weights


@app.post("/upload_weights/")
async def upload_weights(client_id: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    os.makedirs("client_uploads", exist_ok=True)
    path = f"client_uploads/{client_id}_weights.pth"

    with open(path, "wb") as f:
        f.write(contents)

    try:
        weights_state_dict = torch.load(path)
        if not isinstance(weights_state_dict, dict):
            raise ValueError("Uploaded file does not contain a valid PyTorch state_dict.")
        client_weights[client_id] = weights_state_dict
        print(f"Successfully loaded weights for {client_id}. Total clients received: {len(client_weights)}/{NUM_CLIENTS}")
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=500, detail=f"Error loading weights for {client_id}: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)

    if len(client_weights) >= NUM_CLIENTS:
        print(f"Received weights from all {NUM_CLIENTS} clients. Starting aggregation.")
        weights_list = list(client_weights.values())
        aggregated_state_dict = aggregate_weights(weights_list)

        if aggregated_state_dict is None:
            message = "Aggregation failed (e.g., empty weights list or key mismatch). Waiting for new uploads from all clients."
            print(message)
            client_weights.clear() 
            return {"message": message}

        model = SignalClassifierModel()
        try:
            model.load_state_dict(aggregated_state_dict)
            save_global_model(model)
            message = f"Weights received from clients: {', '.join(client_weights.keys())}. Aggregation done and global model updated."
            print(message)
        except RuntimeError as e:
            message = f"Error loading aggregated weights into model: {e}. Global model not updated. This often means a mismatch in model architecture (N_FEATURES, HIDDEN_LAYER_SIZE, NUM_MODULATION_TYPES) between client and server, or corrupted aggregated weights."
            print(message)
            
        
        client_weights.clear()
        return {"message": message}
    else:
        message = f"Weights from {client_id} uploaded. Waiting for {NUM_CLIENTS - len(client_weights)} more client(s)..."
        print(message)
        return {"message": message}


@app.get("/get_global_weights/")
def get_global_weights():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print(f"Global model at {GLOBAL_MODEL_PATH} doesn't exist. Initializing a new one.")
        model = SignalClassifierModel()
        save_global_model(model)
    
    if not os.path.exists(GLOBAL_MODEL_PATH): 
        print(f"Critical Error: Global model file {GLOBAL_MODEL_PATH} not found and could not be initialized.")
        raise HTTPException(status_code=500, detail="Global model file system error during get request.")

    return FileResponse(GLOBAL_MODEL_PATH, media_type='application/octet-stream', filename='global_signal_model.pth')

if __name__ == "__main__":
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print("Initializing global model at startup...")
        initial_model = SignalClassifierModel()
        save_global_model(initial_model)
    else:
        print(f"Found existing global model at {GLOBAL_MODEL_PATH} at startup.")
        
    uvicorn.run(app, host="0.0.0.0", port=5000)
