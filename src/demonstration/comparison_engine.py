"""
Comparison engine for federated vs centralized learning using same real-world data.
Provides comprehensive performance analysis and benchmarking.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json

from src.common.interfaces import SignalSample
from src.sdr_client.signal_processing import SignalProcessor
from src.aggregation_server.aggregation import FederatedAggregator


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    test_split: float = 0.2
    validation_split: float = 0.1
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentResults:
    """Results from a training experiment."""
    method_name: str
    final_accuracy: float
    training_time_seconds: float
    communication_cost_mb: float
    memory_usage_mb: float
    convergence_round: int
    accuracy_history: List[float]
    loss_history: List[float]
    per_class_metrics: Dict[str, Dict[str, float]]
    privacy_score: float
    fairness_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class SignalClassificationModel(nn.Module):
    """Neural network model for signal classification."""
    
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class CentralizedComparison:
    """Engine for comparing federated and centralized learning approaches."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.signal_processor = SignalProcessor()
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
    
    def prepare_dataset(self, samples: List[SignalSample]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Prepare signal samples for training."""
        
        self.logger.info(f"Preparing dataset with {len(samples)} samples")
        
        # Extract features and labels
        features = []
        labels = []
        label_names = []
        
        # Create label mapping
        unique_modulations = list(set(sample.modulation_type for sample in samples))
        label_map = {mod: idx for idx, mod in enumerate(unique_modulations)}
        
        for sample in samples:
            try:
                # Extract features using signal processor
                feature_vector = self.signal_processor.extract_features(sample.iq_data)
                
                # Add additional features
                additional_features = [
                    sample.snr,
                    len(sample.iq_data),
                    np.mean(np.abs(sample.iq_data)),
                    np.std(np.abs(sample.iq_data))
                ]
                
                combined_features = np.concatenate([feature_vector, additional_features])
                features.append(combined_features)
                labels.append(label_map[sample.modulation_type])
                
            except Exception as e:
                self.logger.warning(f"Error processing sample: {e}")
                continue
        
        if not features:
            raise ValueError("No valid features extracted from samples")
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(features))
        y = torch.LongTensor(labels)
        
        self.logger.info(f"Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_modulations)} classes")
        
        return X, y, unique_modulations
    
    def train_centralized_model(self, X: torch.Tensor, y: torch.Tensor, 
                              class_names: List[str]) -> ExperimentResults:
        """Train centralized model on all data."""
        
        self.logger.info("Starting centralized training")
        start_time = time.time()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_split, 
            random_state=self.config.random_seed, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.config.validation_split,
            random_state=self.config.random_seed, stratify=y_train
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        # Initialize model
        model = SignalClassificationModel(
            input_size=X.shape[1],
            num_classes=len(class_names)
        ).to(self.config.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        accuracy_history = []
        loss_history = []
        best_val_accuracy = 0
        convergence_round = self.config.num_epochs
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_accuracy = self._evaluate_model(model, val_loader)
            
            accuracy_history.append(val_accuracy)
            loss_history.append(train_loss / len(train_loader))
            
            # Check for convergence
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            elif epoch > 10 and val_accuracy < best_val_accuracy - 0.01:
                convergence_round = epoch
                break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Val Accuracy = {val_accuracy:.4f}")
        
        # Final evaluation
        test_accuracy = self._evaluate_model(model, test_loader)
        per_class_metrics = self._calculate_per_class_metrics(model, test_loader, class_names)
        
        training_time = time.time() - start_time
        
        # Calculate communication cost (data transfer to central server)
        data_size_mb = (X.numel() * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Calculate memory usage
        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        
        results = ExperimentResults(
            method_name="Centralized Learning",
            final_accuracy=test_accuracy,
            training_time_seconds=training_time,
            communication_cost_mb=data_size_mb,
            memory_usage_mb=model_size_mb,
            convergence_round=convergence_round,
            accuracy_history=accuracy_history,
            loss_history=loss_history,
            per_class_metrics=per_class_metrics,
            privacy_score=0.1,  # Low privacy (all data centralized)
            fairness_metrics=self._calculate_fairness_metrics(model, test_loader, class_names),
            metadata={
                "total_samples": len(X),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": len(class_names),
                "feature_dim": X.shape[1]
            }
        )
        
        self.logger.info(f"Centralized training completed: {test_accuracy:.4f} accuracy in {training_time:.2f}s")
        
        return results
    
    def simulate_federated_learning(self, X: torch.Tensor, y: torch.Tensor, 
                                  class_names: List[str], num_clients: int = 10) -> ExperimentResults:
        """Simulate federated learning with multiple clients."""
        
        self.logger.info(f"Starting federated learning simulation with {num_clients} clients")
        start_time = time.time()
        
        # Split data among clients (non-IID distribution)
        client_data = self._create_non_iid_split(X, y, num_clients)
        
        # Global test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_split,
            random_state=self.config.random_seed, stratify=y
        )
        
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        # Initialize global model
        global_model = SignalClassificationModel(
            input_size=X.shape[1],
            num_classes=len(class_names)
        ).to(self.config.device)
        
        # Federated learning parameters
        num_rounds = 20
        clients_per_round = max(1, num_clients // 2)  # 50% participation
        
        accuracy_history = []
        loss_history = []
        communication_cost = 0
        convergence_round = num_rounds
        best_accuracy = 0
        
        for round_num in range(num_rounds):
            self.logger.info(f"Federated round {round_num + 1}/{num_rounds}")
            
            # Select clients for this round
            selected_clients = np.random.choice(num_clients, clients_per_round, replace=False)
            
            # Client updates
            client_models = []
            client_weights = []
            
            for client_id in selected_clients:
                client_X, client_y = client_data[client_id]
                
                if len(client_X) == 0:
                    continue
                
                # Train local model
                local_model = self._train_local_model(
                    global_model, client_X, client_y, local_epochs=5
                )
                
                client_models.append(local_model)
                client_weights.append(len(client_X))
                
                # Calculate communication cost (model parameters)
                model_size_mb = sum(p.numel() * 4 for p in local_model.parameters()) / (1024 * 1024)
                communication_cost += model_size_mb * 2  # Upload + download
            
            # Aggregate models (FedAvg)
            if client_models:
                global_model = self._federated_averaging(client_models, client_weights)
            
            # Evaluate global model
            accuracy = self._evaluate_model(global_model, test_loader)
            accuracy_history.append(accuracy)
            
            # Check for convergence
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            elif round_num > 5 and accuracy < best_accuracy - 0.01:
                convergence_round = round_num
                break
            
            if round_num % 5 == 0:
                self.logger.info(f"Round {round_num}: Global Accuracy = {accuracy:.4f}")
        
        # Final evaluation
        final_accuracy = self._evaluate_model(global_model, test_loader)
        per_class_metrics = self._calculate_per_class_metrics(global_model, test_loader, class_names)
        
        training_time = time.time() - start_time
        
        # Calculate memory usage
        model_size_mb = sum(p.numel() * 4 for p in global_model.parameters()) / (1024 * 1024)
        
        results = ExperimentResults(
            method_name="Federated Learning",
            final_accuracy=final_accuracy,
            training_time_seconds=training_time,
            communication_cost_mb=communication_cost,
            memory_usage_mb=model_size_mb,
            convergence_round=convergence_round,
            accuracy_history=accuracy_history,
            loss_history=[0] * len(accuracy_history),  # Not tracked in federated setting
            per_class_metrics=per_class_metrics,
            privacy_score=0.9,  # High privacy (data stays local)
            fairness_metrics=self._calculate_fairness_metrics(global_model, test_loader, class_names),
            metadata={
                "total_samples": len(X),
                "num_clients": num_clients,
                "clients_per_round": clients_per_round,
                "num_rounds": len(accuracy_history),
                "num_classes": len(class_names),
                "feature_dim": X.shape[1]
            }
        )
        
        self.logger.info(f"Federated learning completed: {final_accuracy:.4f} accuracy in {training_time:.2f}s")
        
        return results
    
    def run_comprehensive_comparison(self, samples: List[SignalSample], 
                                   num_clients: int = 10) -> Dict[str, ExperimentResults]:
        """Run comprehensive comparison between federated and centralized approaches."""
        
        self.logger.info("Starting comprehensive comparison")
        
        # Prepare dataset
        X, y, class_names = self.prepare_dataset(samples)
        
        # Run centralized training
        centralized_results = self.train_centralized_model(X, y, class_names)
        
        # Run federated learning
        federated_results = self.simulate_federated_learning(X, y, class_names, num_clients)
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(
            centralized_results, federated_results
        )
        
        results = {
            "centralized": centralized_results,
            "federated": federated_results,
            "comparison_report": comparison_report
        }
        
        self.logger.info("Comprehensive comparison completed")
        
        return results
    
    def _create_non_iid_split(self, X: torch.Tensor, y: torch.Tensor, 
                            num_clients: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create non-IID data split among clients."""
        
        client_data = []
        num_classes = len(torch.unique(y))
        
        # Sort by labels
        sorted_indices = torch.argsort(y)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Divide into shards
        num_shards = num_clients * 2  # Each client gets 2 shards
        shard_size = len(X_sorted) // num_shards
        
        shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = (i + 1) * shard_size if i < num_shards - 1 else len(X_sorted)
            shards.append((X_sorted[start_idx:end_idx], y_sorted[start_idx:end_idx]))
        
        # Assign shards to clients
        shard_indices = list(range(num_shards))
        np.random.shuffle(shard_indices)
        
        for client_id in range(num_clients):
            # Each client gets 2 random shards
            client_shards = shard_indices[client_id * 2:(client_id + 1) * 2]
            
            client_X = torch.cat([shards[i][0] for i in client_shards])
            client_y = torch.cat([shards[i][1] for i in client_shards])
            
            client_data.append((client_X, client_y))
        
        return client_data
    
    def _train_local_model(self, global_model: nn.Module, X: torch.Tensor, 
                         y: torch.Tensor, local_epochs: int = 5) -> nn.Module:
        """Train local model for one client."""
        
        # Create local model copy
        local_model = SignalClassificationModel(
            input_size=X.shape[1],
            num_classes=global_model.classifier.out_features
        ).to(self.config.device)
        
        # Copy global model weights
        local_model.load_state_dict(global_model.state_dict())
        
        # Create data loader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=min(self.config.batch_size, len(X)), shuffle=True)
        
        # Train locally
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(local_model.parameters(), lr=self.config.learning_rate)
        
        local_model.train()
        for epoch in range(local_epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return local_model
    
    def _federated_averaging(self, client_models: List[nn.Module], 
                           client_weights: List[int]) -> nn.Module:
        """Perform federated averaging of client models."""
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Get global model structure
        global_model = SignalClassificationModel(
            input_size=client_models[0].feature_extractor[0].in_features,
            num_classes=client_models[0].classifier.out_features
        ).to(self.config.device)
        
        # Average parameters
        global_dict = global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            
            for i, model in enumerate(client_models):
                global_dict[key] += model.state_dict()[key] * normalized_weights[i]
        
        global_model.load_state_dict(global_dict)
        
        return global_model
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total
    
    def _calculate_per_class_metrics(self, model: nn.Module, data_loader: DataLoader, 
                                   class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate per-class precision, recall, and F1-score."""
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision[i]) if i < len(precision) else 0.0,
                "recall": float(recall[i]) if i < len(recall) else 0.0,
                "f1_score": float(f1[i]) if i < len(f1) else 0.0,
                "support": int(support[i]) if i < len(support) else 0
            }
        
        return per_class_metrics
    
    def _calculate_fairness_metrics(self, model: nn.Module, data_loader: DataLoader, 
                                  class_names: List[str]) -> Dict[str, float]:
        """Calculate fairness metrics."""
        
        per_class_metrics = self._calculate_per_class_metrics(model, data_loader, class_names)
        
        # Calculate fairness metrics
        f1_scores = [metrics["f1_score"] for metrics in per_class_metrics.values()]
        
        fairness_metrics = {
            "f1_variance": float(np.var(f1_scores)),
            "f1_min_max_ratio": float(min(f1_scores) / max(f1_scores)) if max(f1_scores) > 0 else 0.0,
            "f1_std": float(np.std(f1_scores))
        }
        
        return fairness_metrics
    
    def _generate_comparison_report(self, centralized: ExperimentResults, 
                                  federated: ExperimentResults) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        
        report = {
            "summary": {
                "accuracy_difference": federated.final_accuracy - centralized.final_accuracy,
                "training_time_ratio": federated.training_time_seconds / centralized.training_time_seconds,
                "communication_cost_ratio": federated.communication_cost_mb / max(centralized.communication_cost_mb, 1),
                "privacy_advantage": federated.privacy_score - centralized.privacy_score
            },
            "performance_comparison": {
                "centralized_accuracy": centralized.final_accuracy,
                "federated_accuracy": federated.final_accuracy,
                "accuracy_gap": abs(centralized.final_accuracy - federated.final_accuracy),
                "convergence_comparison": {
                    "centralized_rounds": centralized.convergence_round,
                    "federated_rounds": federated.convergence_round
                }
            },
            "efficiency_comparison": {
                "training_time": {
                    "centralized_seconds": centralized.training_time_seconds,
                    "federated_seconds": federated.training_time_seconds,
                    "speedup_factor": centralized.training_time_seconds / federated.training_time_seconds
                },
                "communication_cost": {
                    "centralized_mb": centralized.communication_cost_mb,
                    "federated_mb": federated.communication_cost_mb,
                    "cost_ratio": federated.communication_cost_mb / max(centralized.communication_cost_mb, 1)
                }
            },
            "privacy_analysis": {
                "centralized_privacy_score": centralized.privacy_score,
                "federated_privacy_score": federated.privacy_score,
                "privacy_improvement": federated.privacy_score - centralized.privacy_score
            },
            "fairness_analysis": {
                "centralized_fairness": centralized.fairness_metrics,
                "federated_fairness": federated.fairness_metrics
            },
            "recommendations": self._generate_recommendations(centralized, federated)
        }
        
        return report
    
    def _generate_recommendations(self, centralized: ExperimentResults, 
                                federated: ExperimentResults) -> List[str]:
        """Generate recommendations based on comparison results."""
        
        recommendations = []
        
        accuracy_diff = federated.final_accuracy - centralized.final_accuracy
        
        if accuracy_diff > 0.05:
            recommendations.append("Federated learning shows superior accuracy. Consider federated approach.")
        elif accuracy_diff < -0.05:
            recommendations.append("Centralized learning shows superior accuracy. Evaluate if privacy trade-off is acceptable.")
        else:
            recommendations.append("Both approaches show similar accuracy. Choose based on privacy and infrastructure requirements.")
        
        if federated.communication_cost_mb > centralized.communication_cost_mb * 2:
            recommendations.append("High communication cost in federated learning. Consider model compression or less frequent updates.")
        
        if federated.training_time_seconds > centralized.training_time_seconds * 1.5:
            recommendations.append("Federated learning takes significantly longer. Consider parallel training or fewer rounds.")
        
        if federated.privacy_score > 0.8:
            recommendations.append("Federated learning provides strong privacy preservation benefits.")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save comparison results to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        
        for key, value in results.items():
            if isinstance(value, ExperimentResults):
                serializable_results[key] = {
                    "method_name": value.method_name,
                    "final_accuracy": value.final_accuracy,
                    "training_time_seconds": value.training_time_seconds,
                    "communication_cost_mb": value.communication_cost_mb,
                    "memory_usage_mb": value.memory_usage_mb,
                    "convergence_round": value.convergence_round,
                    "accuracy_history": value.accuracy_history,
                    "loss_history": value.loss_history,
                    "per_class_metrics": value.per_class_metrics,
                    "privacy_score": value.privacy_score,
                    "fairness_metrics": value.fairness_metrics,
                    "metadata": value.metadata
                }
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        
        return filename