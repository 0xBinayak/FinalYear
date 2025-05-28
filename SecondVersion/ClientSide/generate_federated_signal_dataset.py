import torch
import numpy as np
from scipy.stats import skew, kurtosis
import os

N_FEATURES = 16
NUM_MODULATION_TYPES = 3
MOD_CLASSES = {0: 'BPSK', 1: 'QPSK', 2: '8PSK'}
NUM_SYMBOLS_PER_SIGNAL = 512 


def generate_symbols(mod_type, num_symbols=NUM_SYMBOLS_PER_SIGNAL):
    if mod_type == 0:  # BPSK
        symbols = np.random.choice([-1, 1], num_symbols).astype(np.complex64)
    elif mod_type == 1:  # QPSK
        real_part = np.random.choice([-1, 1], num_symbols)
        imag_part = np.random.choice([-1, 1], num_symbols)
        symbols = (real_part + 1j * imag_part).astype(np.complex64) / np.sqrt(2)
    elif mod_type == 2:  # 8PSK
        angles = np.random.choice(np.arange(0, 8) * (2 * np.pi / 8), num_symbols)
        symbols = np.exp(1j * angles).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported modulation type index: {mod_type}")
    return symbols

def add_awgn_noise(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    
    if snr_linear == 0: 
        noise_power = np.inf
    else:
        noise_power = signal_power / snr_linear
    
    
    if signal_power == 0 and noise_power == np.inf:
        
        noise_std_dev = 1.0 
    elif noise_power == np.inf:
        noise_std_dev = 10.0 * np.sqrt(1.0) 
    elif noise_power == 0 : 
        noise_std_dev = 0.0
    else:
        noise_std_dev = np.sqrt(noise_power / 2.0)

    noise = noise_std_dev * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise.astype(np.complex64)

def extract_features(signal_iq):
    features = []
    magnitude = np.abs(signal_iq)
    
    if len(magnitude) < 2: 
        print("Warning: Signal length too short for feature extraction, returning zeros.")
        return torch.zeros(N_FEATURES, dtype=torch.float32) 

    features.extend([np.mean(magnitude), np.std(magnitude), skew(magnitude), kurtosis(magnitude, fisher=True, bias=False)])
    
    non_zero_signal_iq = signal_iq[magnitude > 1e-9] 
    if len(non_zero_signal_iq) > 1: 
        phase = np.angle(non_zero_signal_iq)
        features.extend([np.mean(phase), np.std(phase), skew(phase), kurtosis(phase, fisher=True, bias=False)])
    else: 
        features.extend([0.0, 0.0, 0.0, 0.0]) 
    
    real_part = np.real(signal_iq)
    features.extend([np.mean(real_part), np.std(real_part), skew(real_part), kurtosis(real_part, fisher=True, bias=False)])
    
    imag_part = np.imag(signal_iq)
    features.extend([np.mean(imag_part), np.std(imag_part), skew(imag_part), kurtosis(imag_part, fisher=True, bias=False)])
    
    if len(features) != N_FEATURES:
        print(f"Warning: Feature extraction generated {len(features)} features, expected {N_FEATURES}. Adjusting.")
        if len(features) > N_FEATURES:
            features = features[:N_FEATURES] 
        else:
            features.extend([0.0] * (N_FEATURES - len(features))) 
            
    return torch.tensor(features, dtype=torch.float32)

def create_dataset_for_client(num_signals_per_class, client_id_str, snr_params):
    all_features = []
    all_labels = []
    mean_snr, std_dev_snr = snr_params

    print(f"Generating data for {client_id_str} with mean SNR {mean_snr:.2f} dB, std dev {std_dev_snr:.2f} dB")

    for mod_idx in range(NUM_MODULATION_TYPES):
        for _ in range(num_signals_per_class):
            current_snr_db = np.random.normal(mean_snr, std_dev_snr)
            symbols = generate_symbols(mod_idx)
            noisy_signal = add_awgn_noise(symbols, current_snr_db)
            features = extract_features(noisy_signal)
            
            all_features.append(features)
            all_labels.append(mod_idx)
    
    if not all_features: 
        print(f"Warning: No features generated for {client_id_str}. Returning empty tensors.")
        return torch.empty(0, N_FEATURES), torch.empty(0, dtype=torch.long)

    indices = np.arange(len(all_features))
    np.random.shuffle(indices)
    
    shuffled_features = torch.stack(all_features)[indices]
    shuffled_labels = torch.tensor(all_labels, dtype=torch.long)[indices]

    return shuffled_features, shuffled_labels

def save_data(features, labels, base_filename, data_dir="federated_data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    features_path = os.path.join(data_dir, f"{base_filename}_features.pt")
    labels_path = os.path.join(data_dir, f"{base_filename}_labels.pt")
    torch.save(features, features_path)
    torch.save(labels, labels_path)
    print(f"Saved {len(labels)} samples for {base_filename} to {features_path} and {labels_path}")

# --- Main Data Generation Logic ---
if __name__ == "__main__":
    np.random.seed(42) 
    torch.manual_seed(42)

    num_train_signals_per_class_client = 250 
    num_test_signals_per_class = 150       
    num_inference_signals_total = 20       

    client1_snr = (12, 3) 
    client2_snr = (7, 4)  
    client3_snr = (15, 2) 

    test_set_snr_range = (5, 18) 

    c1_feats, c1_labels = create_dataset_for_client(num_train_signals_per_class_client, "ClientPC1", client1_snr)
    save_data(c1_feats, c1_labels, "clientpc1_train_data") 
    c2_feats, c2_labels = create_dataset_for_client(num_train_signals_per_class_client, "ClientPC2", client2_snr)
    save_data(c2_feats, c2_labels, "clientpc2_train_data") 
    c3_feats, c3_labels = create_dataset_for_client(num_train_signals_per_class_client, "ClientPC3", client3_snr)
    save_data(c3_feats, c3_labels, "clientpc3_train_data") 

    
    test_features_list = []
    test_labels_list = []
    print(f"Generating global test data with SNRs between {test_set_snr_range[0]} and {test_set_snr_range[1]} dB")
    for mod_idx in range(NUM_MODULATION_TYPES):
        for _ in range(num_test_signals_per_class):
            current_snr_db = np.random.uniform(test_set_snr_range[0], test_set_snr_range[1])
            symbols = generate_symbols(mod_idx)
            noisy_signal = add_awgn_noise(symbols, current_snr_db)
            features = extract_features(noisy_signal)
            test_features_list.append(features)
            test_labels_list.append(mod_idx)
            
    if test_features_list: 
        test_indices = np.arange(len(test_features_list))
        np.random.shuffle(test_indices)
        global_test_features = torch.stack(test_features_list)[test_indices]
        global_test_labels = torch.tensor(test_labels_list, dtype=torch.long)[test_indices]
        save_data(global_test_features, global_test_labels, "global_test_data")
    else:
        print("Warning: No test features generated.")

    # 5. Generate Inference Data
    inference_features_list = []
    inference_labels_list = []
    inference_params = [
        (0, 20), (1, 20), 
        (2, 15), (0, 15), 
        (1, 10), (2, 10), 
        (0, 5),  (1, 5),   
        (2, 0),  (0, 0)    
    ] 
    if inference_params: 
        actual_inference_params = (inference_params * (num_inference_signals_total // len(inference_params) + 1))[:num_inference_signals_total]
    else:
        actual_inference_params = []

    print(f"Generating inference data (total {len(actual_inference_params)} signals)...")
    for mod_idx, snr_db in actual_inference_params:
        symbols = generate_symbols(mod_idx)
        noisy_signal = add_awgn_noise(symbols, snr_db)
        features = extract_features(noisy_signal)
        inference_features_list.append(features)
        inference_labels_list.append(mod_idx)
    
    if inference_features_list: 
        inference_features = torch.stack(inference_features_list)
        inference_labels = torch.tensor(inference_labels_list, dtype=torch.long)
        save_data(inference_features, inference_labels, "inference_data_batch")
    else:
        print("Warning: No inference features generated.")

    print("\n--- Dataset Generation Complete ---")
    if 'c1_labels' in locals() and c1_labels is not None: print(f"Client 1 training data: {len(c1_labels)} samples")
    if 'c2_labels' in locals() and c2_labels is not None: print(f"Client 2 training data: {len(c2_labels)} samples")
    if 'c3_labels' in locals() and c3_labels is not None: print(f"Client 3 training data: {len(c3_labels)} samples")
    if 'global_test_labels' in locals() and global_test_labels is not None: print(f"Global test data: {len(global_test_labels)} samples")
    if 'inference_labels' in locals() and inference_labels is not None: print(f"Inference data batch: {len(inference_labels)} samples")
    print(f"\nAll data saved in '{os.path.join(os.getcwd(), 'federated_data')}' directory.")
