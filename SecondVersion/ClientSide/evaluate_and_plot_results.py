
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import seaborn as sns 


N_FEATURES = 16
NUM_MODULATION_TYPES = 3
MOD_CLASSES = {0: 'BPSK', 1: 'QPSK', 2: '8PSK'}
CLASS_NAMES = [MOD_CLASSES[i] for i in range(NUM_MODULATION_TYPES)]


HIDDEN_LAYER_SIZE = 128  
NUM_LOCAL_EPOCHS_FOR_PURELY_LOCAL = 10 
LEARNING_RATE_FOR_PURELY_LOCAL = 0.001 

DATA_DIR = "federated_data" 
MODEL_DIR = "." 


LOCAL_MODEL_PATHS = {
    f"Client 1 (Local H{HIDDEN_LAYER_SIZE} E{NUM_LOCAL_EPOCHS_FOR_PURELY_LOCAL} LR{LEARNING_RATE_FOR_PURELY_LOCAL})": os.path.join(MODEL_DIR, f"client_client_pc1_purely_local_model_H{HIDDEN_LAYER_SIZE}_E{NUM_LOCAL_EPOCHS_FOR_PURELY_LOCAL}_LR{LEARNING_RATE_FOR_PURELY_LOCAL}.pth"),
    f"Client 2 (Local H{HIDDEN_LAYER_SIZE} E{NUM_LOCAL_EPOCHS_FOR_PURELY_LOCAL} LR{LEARNING_RATE_FOR_PURELY_LOCAL})": os.path.join(MODEL_DIR, f"client_client_pc2_purely_local_model_H{HIDDEN_LAYER_SIZE}_E{NUM_LOCAL_EPOCHS_FOR_PURELY_LOCAL}_LR{LEARNING_RATE_FOR_PURELY_LOCAL}.pth"),
    f"Client 3 (Local H{HIDDEN_LAYER_SIZE} E{NUM_LOCAL_EPOCHS_FOR_PURELY_LOCAL} LR{LEARNING_RATE_FOR_PURELY_LOCAL})": os.path.join(MODEL_DIR, f"client_client_pc3_purely_local_model_H{HIDDEN_LAYER_SIZE}_E{NUM_LOCAL_EPOCHS_FOR_PURELY_LOCAL}_LR{LEARNING_RATE_FOR_PURELY_LOCAL}.pth"),
}

GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_global_model_used_for_inference.pth") 


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

def load_dataset(base_filename):
    features_path = os.path.join(DATA_DIR, f"{base_filename}_features.pt")
    labels_path = os.path.join(DATA_DIR, f"{base_filename}_labels.pt")
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Error: Dataset files not found for {base_filename} in {DATA_DIR}")
        print(f"Expected: {features_path} and {labels_path}")
        return None, None
    features = torch.load(features_path)
    labels = torch.load(labels_path)
    return features, labels

def evaluate_model(model, features, labels):
    model.eval() 
    all_preds = []
    all_true = []
    with torch.no_grad():
        outputs = model(features)
        _, predicted_indices = torch.max(outputs, 1)
        all_preds.extend(predicted_indices.cpu().numpy())
        all_true.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_true, all_preds)
    return accuracy, all_true, all_preds

def plot_confusion_matrix_func(cm, class_names, title='Confusion Matrix'): 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    


if __name__ == "__main__":
    print("--- Starting Model Evaluation ---")
    print(f"--- Expecting models with Hidden Layer Size: {HIDDEN_LAYER_SIZE} ---")


    # 1. Load Global Test Data
    test_features, test_labels = load_dataset("global_test_data")
    if test_features is None:
        print("Exiting due to missing test data.")
        exit()
    print(f"Loaded global test data: {len(test_labels)} samples.")

    # 2. Load Inference Data
    inference_features, inference_labels_actual_indices_eval = load_dataset("inference_data_batch") 
    if inference_features is None:
        print("Exiting due to missing inference data.")
        exit()
    print(f"Loaded inference data: {len(inference_labels_actual_indices_eval)} samples.")

    model_accuracies = {}
    model_predictions_on_test = {}
    plots_to_show = []

    
    for model_name, model_path_val in LOCAL_MODEL_PATHS.items():
        if not os.path.exists(model_path_val):
            print(f"Warning: Local model file not found: {model_path_val}. Skipping.")
            model_accuracies[model_name] = 0.0 
            continue
        
        print(f"\nEvaluating {model_name}...")
        local_model = SignalClassifierModel() 
        try:
            local_model.load_state_dict(torch.load(model_path_val))
            accuracy, _, preds = evaluate_model(local_model, test_features, test_labels)
            model_accuracies[model_name] = accuracy
            model_predictions_on_test[model_name] = preds
            print(f"  Accuracy on Global Test Set: {accuracy:.4f}")
        except RuntimeError as e:
            print(f"  Error loading local model {model_name}: {e}. This might be due to architecture mismatch (e.g., HIDDEN_LAYER_SIZE).")
            model_accuracies[model_name] = 0.0
        except Exception as e:
            print(f"  Generic error loading or evaluating {model_name}: {e}")
            model_accuracies[model_name] = 0.0


    
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print(f"Warning: Global model file not found: {GLOBAL_MODEL_PATH}. Make sure you downloaded it from the Pi.")
        model_accuracies["Federated Global Model"] = 0.0
    else:
        print(f"\nEvaluating Federated Global Model (Path: {GLOBAL_MODEL_PATH})...")
        global_model_eval = SignalClassifierModel() 
        try:
            global_model_eval.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
            accuracy, true_test_labels_eval, global_preds_test_eval = evaluate_model(global_model_eval, test_features, test_labels) 
            model_accuracies["Federated Global Model"] = accuracy
            model_predictions_on_test["Federated Global Model"] = global_preds_test_eval
            print(f"  Accuracy on Global Test Set: {accuracy:.4f}")

            print("\nClassification Report for Global Model on Test Data:")
            print(classification_report(true_test_labels_eval, global_preds_test_eval, target_names=CLASS_NAMES, zero_division=0))
            cm_global_test = confusion_matrix(true_test_labels_eval, global_preds_test_eval)
            plot_confusion_matrix_func(cm_global_test, CLASS_NAMES, title='Global Model - Confusion Matrix on Test Data')
            plots_to_show.append(plt.gcf()) 

        except RuntimeError as e:
            print(f"  Error loading global model: {e}. This might be due to architecture mismatch (e.g., HIDDEN_LAYER_SIZE).")
            print(f"  Ensure HIDDEN_LAYER_SIZE ({HIDDEN_LAYER_SIZE}) in this script matches the one used for training the global model.")
            model_accuracies["Federated Global Model"] = 0.0
        except Exception as e:
            print(f"  Generic error loading or evaluating Federated Global Model: {e}")
            model_accuracies["Federated Global Model"] = 0.0


    
    print("\n--- Performing Inference on Inference Batch (in sets of two) using Global Model ---")
    if "Federated Global Model" in model_accuracies and model_accuracies["Federated Global Model"] > 0 and os.path.exists(GLOBAL_MODEL_PATH):
        
        if 'global_model_eval' not in locals() or model_accuracies["Federated Global Model"] == 0.0 : 
            print("Re-attempting to load global model for inference as it wasn't ready.")
            global_model_for_inference = SignalClassifierModel()
            try:
                global_model_for_inference.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
            except Exception as e:
                print(f"Could not load global model for inference: {e}")
                global_model_for_inference = None 
        else:
            global_model_for_inference = global_model_eval 

        if global_model_for_inference:
            global_model_for_inference.eval() 
            all_predicted_indices_inference = []
            with torch.no_grad():
                outputs_inference = global_model_for_inference(inference_features)
                _, predicted_indices_inf = torch.max(outputs_inference, 1)
                all_predicted_indices_inference = predicted_indices_inf.cpu().numpy()

            num_correct_inf = 0
            for i in range(0, len(inference_features), 2):
                print(f"\nInference Set {i//2 + 1}:")
                for j in range(2): 
                    current_index = i + j
                    if current_index >= len(inference_features):
                        break 

                    actual_label_idx = inference_labels_actual_indices_eval[current_index].item()
                    pred_label_idx = all_predicted_indices_inference[current_index]
                    
                    actual_name = MOD_CLASSES.get(actual_label_idx, "Unknown")
                    pred_name = MOD_CLASSES.get(pred_label_idx, "Unknown")
                    
                    correct_str = "CORRECT" if actual_label_idx == pred_label_idx else "INCORRECT"
                    if actual_label_idx == pred_label_idx:
                        num_correct_inf +=1

                    print(f"  Sample {current_index + 1}: Actual='{actual_name}' (Idx:{actual_label_idx}), Predicted='{pred_name}' (Idx:{pred_label_idx}) -> {correct_str}")
            
            if len(inference_features) > 0:
                inference_accuracy = num_correct_inf / len(inference_features)
                print(f"\nAccuracy on this inference batch: {num_correct_inf}/{len(inference_features)} = {inference_accuracy:.4f}")
            else:
                print("No samples in the inference batch to evaluate.")
        else:
            print("Global model not available for inference.")
            
    else:
        print("Skipping inference batch because the global model was not loaded or evaluated successfully.")

    
    print("\n--- Plotting Accuracy Comparison ---")
    if model_accuracies: 
        names = list(model_accuracies.keys())
        accuracies = list(model_accuracies.values())

        plt.figure(figsize=(12, 7)) 
        bars = plt.bar(names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold']) 
        plt.xlabel("Model Type")
        plt.ylabel("Accuracy on Global Test Set")
        plt.title("Comparison of Model Accuracies")
        plt.ylim(0, max(1.05, max(accuracies) + 0.1 if accuracies else 1.05) )
        plt.xticks(rotation=20, ha="right") 
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plots_to_show.append(plt.gcf())
    else:
        print("No model accuracies recorded to plot.")
    
    
    if plots_to_show:
        print(f"\nDisplaying {len(plots_to_show)} plot(s)...")
        plt.show()
    else:
        print("No plots to display.")
        
    print("\n--- Evaluation Complete ---")
