"""
Privacy and security mechanisms for federated learning
"""
import logging
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import hmac
import secrets

from ..common.interfaces import ModelUpdate
from ..common.config import AppConfig

logger = logging.getLogger(__name__)


class DifferentialPrivacyMechanism:
    """Differential privacy implementation for federated learning"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.epsilon = config.privacy.epsilon
        self.delta = config.privacy.delta
        self.noise_multiplier = config.privacy.noise_multiplier
        self.client_budgets: Dict[str, float] = defaultdict(float)
        self.global_budget_used = 0.0
        self.max_global_budget = 10.0  # Total privacy budget
    
    def add_noise_to_weights(self, weights: bytes, sensitivity: float = 1.0) -> bytes:
        """Add differential privacy noise to model weights"""
        try:
            # Deserialize weights
            model_weights = pickle.loads(weights)
            
            # Add noise based on weight format
            if isinstance(model_weights, dict):
                noisy_weights = self._add_noise_dict(model_weights, sensitivity)
            elif isinstance(model_weights, list):
                noisy_weights = self._add_noise_list(model_weights, sensitivity)
            elif isinstance(model_weights, np.ndarray):
                noisy_weights = self._add_noise_array(model_weights, sensitivity)
            else:
                logger.warning(f"Unsupported weight format for DP: {type(model_weights)}")
                return weights
            
            return pickle.dumps(noisy_weights)
        
        except Exception as e:
            logger.error(f"Failed to add DP noise: {e}")
            return weights
    
    def _add_noise_dict(self, weights: Dict, sensitivity: float) -> Dict:
        """Add noise to dictionary weights"""
        noisy_weights = {}
        
        for param_name, param_value in weights.items():
            if hasattr(param_value, 'numpy'):
                param_array = param_value.numpy()
            else:
                param_array = np.array(param_value)
            
            # Calculate noise scale
            noise_scale = sensitivity * self.noise_multiplier / self.epsilon
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, param_array.shape)
            noisy_param = param_array + noise
            
            noisy_weights[param_name] = noisy_param
        
        return noisy_weights
    
    def _add_noise_list(self, weights: List, sensitivity: float) -> List:
        """Add noise to list weights"""
        noisy_weights = []
        
        for layer_weights in weights:
            layer_array = np.array(layer_weights)
            noise_scale = sensitivity * self.noise_multiplier / self.epsilon
            noise = np.random.normal(0, noise_scale, layer_array.shape)
            noisy_layer = layer_array + noise
            noisy_weights.append(noisy_layer)
        
        return noisy_weights
    
    def _add_noise_array(self, weights: np.ndarray, sensitivity: float) -> np.ndarray:
        """Add noise to array weights"""
        noise_scale = sensitivity * self.noise_multiplier / self.epsilon
        noise = np.random.normal(0, noise_scale, weights.shape)
        return weights + noise
    
    def check_privacy_budget(self, client_id: str, requested_budget: float) -> bool:
        """Check if client has sufficient privacy budget"""
        current_budget = self.client_budgets[client_id]
        max_client_budget = 5.0  # Maximum budget per client
        
        if current_budget + requested_budget > max_client_budget:
            logger.warning(f"Client {client_id} exceeds privacy budget")
            return False
        
        if self.global_budget_used + requested_budget > self.max_global_budget:
            logger.warning("Global privacy budget exceeded")
            return False
        
        return True
    
    def consume_privacy_budget(self, client_id: str, budget_used: float):
        """Consume privacy budget for client"""
        self.client_budgets[client_id] += budget_used
        self.global_budget_used += budget_used
        
        logger.debug(f"Privacy budget consumed - Client {client_id}: {budget_used}, "
                    f"Total: {self.client_budgets[client_id]}")
    
    def get_privacy_budget_status(self) -> Dict[str, Any]:
        """Get privacy budget status"""
        return {
            'global_budget_used': self.global_budget_used,
            'global_budget_remaining': self.max_global_budget - self.global_budget_used,
            'client_budgets': dict(self.client_budgets),
            'epsilon': self.epsilon,
            'delta': self.delta
        }


class SecureMultiPartyComputation:
    """Secure multi-party computation for sensitive aggregation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.secret_shares: Dict[str, List[bytes]] = {}
        self.reconstruction_threshold = 3  # Minimum shares needed
    
    def create_secret_shares(self, weights: bytes, num_shares: int = 5) -> List[bytes]:
        """Create secret shares of model weights using Shamir's secret sharing"""
        try:
            # Simple additive secret sharing for demonstration
            # In production, use proper Shamir's secret sharing
            
            model_weights = pickle.loads(weights)
            shares = []
            
            # Generate random shares
            for i in range(num_shares - 1):
                share = self._generate_random_share(model_weights)
                shares.append(pickle.dumps(share))
            
            # Last share is computed to maintain sum
            final_share = self._compute_final_share(model_weights, shares)
            shares.append(pickle.dumps(final_share))
            
            return shares
        
        except Exception as e:
            logger.error(f"Failed to create secret shares: {e}")
            return [weights]  # Fallback to original weights
    
    def reconstruct_from_shares(self, shares: List[bytes]) -> bytes:
        """Reconstruct weights from secret shares"""
        try:
            if len(shares) < self.reconstruction_threshold:
                raise ValueError(f"Insufficient shares: {len(shares)} < {self.reconstruction_threshold}")
            
            # Deserialize shares
            share_weights = [pickle.loads(share) for share in shares]
            
            # Simple additive reconstruction
            if isinstance(share_weights[0], dict):
                reconstructed = self._reconstruct_dict_shares(share_weights)
            elif isinstance(share_weights[0], list):
                reconstructed = self._reconstruct_list_shares(share_weights)
            else:
                reconstructed = self._reconstruct_array_shares(share_weights)
            
            return pickle.dumps(reconstructed)
        
        except Exception as e:
            logger.error(f"Failed to reconstruct from shares: {e}")
            return shares[0]  # Fallback to first share
    
    def _generate_random_share(self, weights: Any) -> Any:
        """Generate random share with same structure as weights"""
        if isinstance(weights, dict):
            return {k: np.random.randn(*np.array(v).shape) for k, v in weights.items()}
        elif isinstance(weights, list):
            return [np.random.randn(*np.array(w).shape) for w in weights]
        else:
            return np.random.randn(*np.array(weights).shape)
    
    def _compute_final_share(self, original_weights: Any, shares: List[bytes]) -> Any:
        """Compute final share to maintain sum"""
        share_weights = [pickle.loads(share) for share in shares]
        
        if isinstance(original_weights, dict):
            final_share = {}
            for param_name in original_weights.keys():
                param_sum = sum(np.array(share[param_name]) for share in share_weights)
                final_share[param_name] = np.array(original_weights[param_name]) - param_sum
            return final_share
        elif isinstance(original_weights, list):
            final_share = []
            for i, layer in enumerate(original_weights):
                layer_sum = sum(np.array(share[i]) for share in share_weights)
                final_share.append(np.array(layer) - layer_sum)
            return final_share
        else:
            weights_sum = sum(np.array(share) for share in share_weights)
            return np.array(original_weights) - weights_sum
    
    def _reconstruct_dict_shares(self, shares: List[Dict]) -> Dict:
        """Reconstruct dictionary weights from shares"""
        reconstructed = {}
        param_names = shares[0].keys()
        
        for param_name in param_names:
            param_sum = sum(np.array(share[param_name]) for share in shares)
            reconstructed[param_name] = param_sum
        
        return reconstructed
    
    def _reconstruct_list_shares(self, shares: List[List]) -> List:
        """Reconstruct list weights from shares"""
        reconstructed = []
        
        for i in range(len(shares[0])):
            layer_sum = sum(np.array(share[i]) for share in shares)
            reconstructed.append(layer_sum)
        
        return reconstructed
    
    def _reconstruct_array_shares(self, shares: List[np.ndarray]) -> np.ndarray:
        """Reconstruct array weights from shares"""
        return sum(np.array(share) for share in shares)


class AnomalyDetector:
    """Anomaly detection for adversarial attack identification"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.update_history: List[Dict[str, Any]] = []
        self.client_profiles: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def detect_anomalies(self, updates: List[ModelUpdate]) -> List[str]:
        """Detect anomalous updates that might be adversarial"""
        anomalous_clients = []
        
        try:
            # Extract features from updates
            update_features = []
            for update in updates:
                features = self._extract_update_features(update)
                update_features.append((update.client_id, features))
            
            # Detect statistical outliers
            outliers = self._detect_statistical_outliers(update_features)
            anomalous_clients.extend(outliers)
            
            # Detect behavioral anomalies
            behavioral_anomalies = self._detect_behavioral_anomalies(updates)
            anomalous_clients.extend(behavioral_anomalies)
            
            # Update client profiles
            self._update_client_profiles(updates)
            
            # Store update history
            self.update_history.append({
                'timestamp': datetime.utcnow(),
                'num_updates': len(updates),
                'anomalous_clients': anomalous_clients
            })
            
            if anomalous_clients:
                logger.warning(f"Detected anomalous clients: {anomalous_clients}")
            
            return list(set(anomalous_clients))  # Remove duplicates
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _extract_update_features(self, update: ModelUpdate) -> Dict[str, float]:
        """Extract features from model update for anomaly detection"""
        try:
            weights = pickle.loads(update.model_weights)
            
            # Calculate weight statistics
            if isinstance(weights, dict):
                weight_values = np.concatenate([np.array(w).flatten() for w in weights.values()])
            elif isinstance(weights, list):
                weight_values = np.concatenate([np.array(w).flatten() for w in weights])
            else:
                weight_values = np.array(weights).flatten()
            
            features = {
                'weight_mean': float(np.mean(weight_values)),
                'weight_std': float(np.std(weight_values)),
                'weight_max': float(np.max(weight_values)),
                'weight_min': float(np.min(weight_values)),
                'weight_norm': float(np.linalg.norm(weight_values)),
                'training_loss': update.training_metrics.get('loss', 0.0),
                'training_accuracy': update.training_metrics.get('accuracy', 0.0),
                'computation_time': update.computation_time,
                'data_size': update.data_statistics.get('num_samples', 0)
            }
            
            return features
        
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _detect_statistical_outliers(self, update_features: List[Tuple[str, Dict]]) -> List[str]:
        """Detect statistical outliers using z-score"""
        if len(update_features) < 3:
            return []  # Need at least 3 samples for meaningful statistics
        
        outliers = []
        
        # Extract feature values
        feature_names = update_features[0][1].keys()
        
        for feature_name in feature_names:
            values = [features[feature_name] for _, features in update_features]
            
            if len(set(values)) == 1:  # All values are the same
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                continue
            
            # Calculate z-scores
            for client_id, features in update_features:
                z_score = abs((features[feature_name] - mean_val) / std_val)
                
                if z_score > self.anomaly_threshold:
                    outliers.append(client_id)
                    logger.debug(f"Statistical outlier detected: {client_id}, "
                               f"feature: {feature_name}, z-score: {z_score:.2f}")
        
        return outliers
    
    def _detect_behavioral_anomalies(self, updates: List[ModelUpdate]) -> List[str]:
        """Detect behavioral anomalies based on client history"""
        anomalies = []
        
        for update in updates:
            client_id = update.client_id
            client_profile = self.client_profiles.get(client_id, {})
            
            # Check for sudden changes in behavior
            if 'avg_loss' in client_profile:
                current_loss = update.training_metrics.get('loss', 0.0)
                avg_loss = client_profile['avg_loss']
                
                # Detect sudden improvement (possible model replacement attack)
                if current_loss < avg_loss * 0.1 and avg_loss > 0.1:
                    anomalies.append(client_id)
                    logger.debug(f"Behavioral anomaly: {client_id} sudden loss improvement")
                
                # Detect sudden degradation (possible poisoning)
                if current_loss > avg_loss * 5 and current_loss > 1.0:
                    anomalies.append(client_id)
                    logger.debug(f"Behavioral anomaly: {client_id} sudden loss degradation")
        
        return anomalies
    
    def _update_client_profiles(self, updates: List[ModelUpdate]):
        """Update client behavioral profiles"""
        for update in updates:
            client_id = update.client_id
            profile = self.client_profiles[client_id]
            
            # Update running averages
            loss = update.training_metrics.get('loss', 0.0)
            accuracy = update.training_metrics.get('accuracy', 0.0)
            
            if 'avg_loss' not in profile:
                profile['avg_loss'] = loss
                profile['avg_accuracy'] = accuracy
                profile['update_count'] = 1
            else:
                count = profile['update_count']
                profile['avg_loss'] = (profile['avg_loss'] * count + loss) / (count + 1)
                profile['avg_accuracy'] = (profile['avg_accuracy'] * count + accuracy) / (count + 1)
                profile['update_count'] = count + 1
            
            profile['last_update'] = datetime.utcnow()
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get anomaly detection report"""
        recent_anomalies = [
            entry for entry in self.update_history[-10:]  # Last 10 rounds
            if entry['anomalous_clients']
        ]
        
        return {
            'total_rounds_monitored': len(self.update_history),
            'recent_anomalies': recent_anomalies,
            'client_profiles': dict(self.client_profiles),
            'anomaly_threshold': self.anomaly_threshold
        }


class AuditLogger:
    """Audit logging and compliance reporting"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.audit_log: List[Dict[str, Any]] = []
        self.compliance_events: List[Dict[str, Any]] = []
    
    def log_event(self, event_type: str, client_id: str, details: Dict[str, Any]):
        """Log audit event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'client_id': client_id,
            'details': details,
            'hash': self._calculate_event_hash(event_type, client_id, details)
        }
        
        self.audit_log.append(event)
        
        # Check for compliance events
        if self._is_compliance_event(event_type):
            self.compliance_events.append(event)
        
        logger.info(f"Audit event logged: {event_type} for {client_id}")
    
    def _calculate_event_hash(self, event_type: str, client_id: str, details: Dict[str, Any]) -> str:
        """Calculate cryptographic hash for event integrity"""
        event_data = f"{event_type}:{client_id}:{str(details)}"
        return hashlib.sha256(event_data.encode()).hexdigest()
    
    def _is_compliance_event(self, event_type: str) -> bool:
        """Check if event type requires compliance reporting"""
        compliance_events = [
            'privacy_budget_exceeded',
            'anomaly_detected',
            'client_blocked',
            'data_access',
            'model_export'
        ]
        return event_type in compliance_events
    
    def verify_log_integrity(self) -> bool:
        """Verify audit log integrity"""
        try:
            for event in self.audit_log:
                expected_hash = self._calculate_event_hash(
                    event['event_type'],
                    event['client_id'],
                    event['details']
                )
                
                if event['hash'] != expected_hash:
                    logger.error(f"Audit log integrity violation detected")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Audit log verification failed: {e}")
            return False
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        period_events = [
            event for event in self.compliance_events
            if start_date <= datetime.fromisoformat(event['timestamp']) <= end_date
        ]
        
        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(period_events),
            'event_summary': self._summarize_events(period_events),
            'privacy_violations': [
                event for event in period_events
                if event['event_type'] == 'privacy_budget_exceeded'
            ],
            'security_incidents': [
                event for event in period_events
                if event['event_type'] in ['anomaly_detected', 'client_blocked']
            ],
            'data_access_events': [
                event for event in period_events
                if event['event_type'] == 'data_access'
            ],
            'log_integrity_verified': self.verify_log_integrity()
        }
        
        return report
    
    def _summarize_events(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize events by type"""
        summary = defaultdict(int)
        for event in events:
            summary[event['event_type']] += 1
        return dict(summary)
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self.audit_log[-limit:]


class PrivacySecurityManager:
    """Main privacy and security manager"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.dp_mechanism = DifferentialPrivacyMechanism(config)
        self.smpc = SecureMultiPartyComputation(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.audit_logger = AuditLogger(config)
    
    def apply_privacy_protection(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Apply privacy protection to model updates"""
        protected_updates = []
        
        for update in updates:
            try:
                # Check privacy budget
                if not self.dp_mechanism.check_privacy_budget(update.client_id, update.privacy_budget_used):
                    self.audit_logger.log_event(
                        'privacy_budget_exceeded',
                        update.client_id,
                        {'requested_budget': update.privacy_budget_used}
                    )
                    continue  # Skip this update
                
                # Apply differential privacy
                if self.config.privacy.enable_differential_privacy:
                    protected_weights = self.dp_mechanism.add_noise_to_weights(update.model_weights)
                    
                    # Create protected update
                    protected_update = ModelUpdate(
                        client_id=update.client_id,
                        model_weights=protected_weights,
                        training_metrics=update.training_metrics,
                        data_statistics=update.data_statistics,
                        computation_time=update.computation_time,
                        network_conditions=update.network_conditions,
                        privacy_budget_used=update.privacy_budget_used
                    )
                    
                    # Consume privacy budget
                    self.dp_mechanism.consume_privacy_budget(update.client_id, update.privacy_budget_used)
                    
                    protected_updates.append(protected_update)
                    
                    self.audit_logger.log_event(
                        'privacy_protection_applied',
                        update.client_id,
                        {'epsilon': self.config.privacy.epsilon}
                    )
                else:
                    protected_updates.append(update)
            
            except Exception as e:
                logger.error(f"Privacy protection failed for {update.client_id}: {e}")
                self.audit_logger.log_event(
                    'privacy_protection_error',
                    update.client_id,
                    {'error': str(e)}
                )
        
        return protected_updates
    
    def detect_and_filter_anomalies(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Detect anomalies and filter out suspicious updates"""
        try:
            # Detect anomalies
            anomalous_clients = self.anomaly_detector.detect_anomalies(updates)
            
            # Filter out anomalous updates
            filtered_updates = [
                update for update in updates
                if update.client_id not in anomalous_clients
            ]
            
            # Log anomalies
            for client_id in anomalous_clients:
                self.audit_logger.log_event(
                    'anomaly_detected',
                    client_id,
                    {'detection_method': 'statistical_and_behavioral'}
                )
            
            logger.info(f"Filtered {len(updates) - len(filtered_updates)} anomalous updates")
            return filtered_updates
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return updates  # Return original updates if detection fails
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            'privacy_budget_status': self.dp_mechanism.get_privacy_budget_status(),
            'anomaly_report': self.anomaly_detector.get_anomaly_report(),
            'recent_audit_events': self.audit_logger.get_audit_log(50),
            'log_integrity_verified': self.audit_logger.verify_log_integrity()
        }
    
    def generate_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate compliance report for recent period"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return self.audit_logger.generate_compliance_report(start_date, end_date)