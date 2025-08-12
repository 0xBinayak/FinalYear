"""
Test suite for federated learning functionality
"""
import numpy as np
import torch
from datetime import datetime
import logging
import tempfile
import os

from .federated_learning import (
    FederatedLearningClient, TrainingConfig, SignalDataset, SignalClassifierModel,
    ModelCompressor, PrivacyEngine, NetworkManager,
    CompressionMethod, PrivacyMethod
)
from .signal_processing import FeatureExtractor, ModulationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_features(num_samples: int = 100) -> tuple:
    """Generate test features and labels"""
    sample_rate = 1e6
    extractor = FeatureExtractor(sample_rate)
    
    features = []
    labels = []
    
    # Generate different signal types
    modulation_types = [0, 1, 2, 3, 4]  # BPSK, QPSK, 8PSK, 16QAM, OFDM
    
    for i in range(num_samples):
        # Generate random signal
        signal_length = 1024
        mod_type = np.random.choice(modulation_types)
        
        if mod_type == 0:  # BPSK
            symbols = np.random.choice([-1, 1], signal_length//4)
            signal = np.repeat(symbols, 4).astype(complex)
        elif mod_type == 1:  # QPSK
            symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], signal_length//4)
            signal = np.repeat(symbols, 4)
        elif mod_type == 2:  # 8PSK
            constellation = [np.exp(1j * 2 * np.pi * k / 8) for k in range(8)]
            symbols = np.random.choice(constellation, signal_length//4)
            signal = np.repeat(symbols, 4)
        elif mod_type == 3:  # 16QAM
            qam_constellation = []
            for i_val in [-3, -1, 1, 3]:
                for q_val in [-3, -1, 1, 3]:
                    qam_constellation.append(i_val + 1j * q_val)
            symbols = np.random.choice(qam_constellation, signal_length//4)
            signal = np.repeat(symbols, 4) / np.sqrt(10)
        else:  # OFDM
            ofdm_data = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], 64)
            ofdm_symbol = np.fft.ifft(ofdm_data)
            signal = np.tile(ofdm_symbol, signal_length // len(ofdm_symbol))[:signal_length]
        
        # Add noise
        noise = 0.1 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        signal = signal + noise
        
        # Extract features
        feature_vector = extractor.extract_features(signal, 900e6)
        features.append(feature_vector)
        labels.append(mod_type)
    
    return features, labels


class TestSignalDataset:
    """Test signal dataset functionality"""
    
    def test_dataset_creation(self):
        """Test dataset creation and data loading"""
        logger.info("Testing signal dataset creation...")
        
        features, labels = generate_test_features(50)
        dataset = SignalDataset(features, labels)
        
        assert len(dataset) == 50
        
        # Test data loading
        feature_tensor, label_tensor = dataset[0]
        
        assert isinstance(feature_tensor, torch.Tensor)
        assert isinstance(label_tensor, torch.Tensor)
        assert feature_tensor.dtype == torch.float32
        assert label_tensor.dtype == torch.long
        
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        logger.info(f"✓ Feature tensor shape: {feature_tensor.shape}")
        
        return True


class TestSignalClassifierModel:
    """Test signal classifier model"""
    
    def test_model_creation(self):
        """Test model creation and forward pass"""
        logger.info("Testing signal classifier model...")
        
        model = SignalClassifierModel(input_size=78, hidden_size=128, num_classes=5)
        
        # Test forward pass
        batch_size = 10
        input_tensor = torch.randn(batch_size, 78)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 5)
        
        # Test parameter count
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model created with {param_count} parameters")
        logger.info(f"✓ Output shape: {output.shape}")
        
        return True
    
    def test_model_training(self):
        """Test basic model training"""
        logger.info("Testing model training...")
        
        model = SignalClassifierModel(input_size=78, hidden_size=64, num_classes=5)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Generate training data
        features, labels = generate_test_features(20)
        dataset = SignalDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
        
        # Training loop
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_labels in dataloader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if initial_loss is None:
                    initial_loss = loss.item()
            
            final_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}, Loss: {final_loss:.4f}")
        
        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss * 1.5  # Allow some variance
        
        logger.info(f"✓ Training completed: {initial_loss:.4f} → {final_loss:.4f}")
        return True


class TestModelCompressor:
    """Test model compression functionality"""
    
    def test_quantization_compression(self):
        """Test quantization compression"""
        logger.info("Testing quantization compression...")
        
        model = SignalClassifierModel(input_size=78, hidden_size=32, num_classes=5)
        compressor = ModelCompressor(CompressionMethod.QUANTIZATION)
        
        # Compress model
        compressed_data, metadata = compressor.compress_model(
            model, TrainingConfig(quantization_bits=8)
        )
        
        assert isinstance(compressed_data, bytes)
        assert metadata['method'] == 'quantization'
        assert metadata['compression_ratio'] < 1.0
        
        # Decompress model
        decompressed_model = SignalClassifierModel(input_size=78, hidden_size=32, num_classes=5)
        decompressed_model = compressor.decompress_model(
            compressed_data, metadata, decompressed_model
        )
        
        # Test that decompressed model works
        test_input = torch.randn(1, 78)
        output = decompressed_model(test_input)
        assert output.shape == (1, 5)
        
        logger.info(f"✓ Compression ratio: {metadata['compression_ratio']:.3f}")
        logger.info(f"✓ Original size: {metadata['original_size']} bytes")
        logger.info(f"✓ Compressed size: {metadata['compressed_size']} bytes")
        
        return True
    
    def test_sparsification_compression(self):
        """Test sparsification compression"""
        logger.info("Testing sparsification compression...")
        
        model = SignalClassifierModel(input_size=78, hidden_size=32, num_classes=5)
        compressor = ModelCompressor(CompressionMethod.SPARSIFICATION)
        
        # Compress model
        compressed_data, metadata = compressor.compress_model(
            model, TrainingConfig(compression_ratio=0.1)
        )
        
        assert isinstance(compressed_data, bytes)
        assert metadata['method'] == 'sparsification'
        assert metadata['compression_ratio'] < 1.0
        
        # Decompress model
        decompressed_model = SignalClassifierModel(input_size=78, hidden_size=32, num_classes=5)
        decompressed_model = compressor.decompress_model(
            compressed_data, metadata, decompressed_model
        )
        
        # Test functionality
        test_input = torch.randn(1, 78)
        output = decompressed_model(test_input)
        assert output.shape == (1, 5)
        
        logger.info(f"✓ Sparsification compression ratio: {metadata['compression_ratio']:.3f}")
        
        return True


class TestPrivacyEngine:
    """Test privacy-preserving mechanisms"""
    
    def test_differential_privacy(self):
        """Test differential privacy"""
        logger.info("Testing differential privacy...")
        
        model = SignalClassifierModel(input_size=78, hidden_size=32, num_classes=5)
        privacy_engine = PrivacyEngine(PrivacyMethod.DIFFERENTIAL_PRIVACY)
        
        # Get original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.clone()
        
        # Apply differential privacy
        config = TrainingConfig(privacy_epsilon=1.0, noise_multiplier=0.1)
        privacy_info = privacy_engine.apply_privacy(model, config)
        
        assert privacy_info['method'] == 'differential_privacy'
        assert privacy_info['epsilon'] == 1.0
        assert privacy_info['privacy_cost'] > 0
        
        # Check that parameters were modified
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, original_params[name]):
                params_changed = True
                break
        
        assert params_changed, "Parameters should be modified by differential privacy"
        
        logger.info(f"✓ Differential privacy applied with ε={privacy_info['epsilon']}")
        logger.info(f"✓ Privacy cost: {privacy_info['privacy_cost']}")
        
        return True


class TestFederatedLearningClient:
    """Test federated learning client"""
    
    def test_client_initialization(self):
        """Test client initialization"""
        logger.info("Testing federated learning client initialization...")
        
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=16,
            local_epochs=2,
            compression_method=CompressionMethod.QUANTIZATION,
            privacy_method=PrivacyMethod.DIFFERENTIAL_PRIVACY
        )
        
        client = FederatedLearningClient(
            client_id="test_client_001",
            server_url="http://localhost:8000",
            config=config
        )
        
        assert client.client_id == "test_client_001"
        assert client.current_round == 0
        assert not client.is_training
        
        # Test model architecture
        assert isinstance(client.model, SignalClassifierModel)
        
        logger.info(f"✓ Client initialized: {client.client_id}")
        logger.info(f"✓ Model parameters: {sum(p.numel() for p in client.model.parameters())}")
        
        return True
    
    def test_local_training(self):
        """Test local model training"""
        logger.info("Testing local model training...")
        
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=8,
            local_epochs=2,
            compression_method=CompressionMethod.QUANTIZATION,
            privacy_method=PrivacyMethod.DIFFERENTIAL_PRIVACY,
            privacy_epsilon=2.0
        )
        
        client = FederatedLearningClient(
            client_id="test_client_002",
            server_url="http://localhost:8000",
            config=config
        )
        
        # Generate training data
        features, labels = generate_test_features(30)
        
        # Train local model
        model_update = client.train_local_model(features, labels)
        
        assert model_update.client_id == "test_client_002"
        assert model_update.round_number == 0
        assert isinstance(model_update.model_weights, bytes)
        assert len(model_update.model_weights) > 0
        
        # Check training metrics
        metrics = model_update.training_metrics
        assert 'training_time' in metrics
        assert 'num_samples' in metrics
        assert 'final_loss' in metrics
        assert metrics['num_samples'] == 30
        
        # Check compression info
        compression_info = model_update.compression_info
        assert compression_info['method'] == 'quantization'
        
        # Check privacy info
        privacy_info = model_update.privacy_info
        assert privacy_info['method'] == 'differential_privacy'
        assert privacy_info['epsilon'] == 2.0
        
        logger.info(f"✓ Local training completed in {metrics['training_time']:.2f}s")
        logger.info(f"✓ Final loss: {metrics['final_loss']:.4f}")
        logger.info(f"✓ Compression ratio: {compression_info['compression_ratio']:.3f}")
        
        return True
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        logger.info("Testing model evaluation...")
        
        config = TrainingConfig(batch_size=8)
        client = FederatedLearningClient(
            client_id="test_client_003",
            server_url="http://localhost:8000",
            config=config
        )
        
        # Generate test data
        features, labels = generate_test_features(20)
        
        # Evaluate model
        metrics = client.evaluate_model(features, labels)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'num_samples' in metrics
        assert metrics['num_samples'] == 20
        assert 0.0 <= metrics['accuracy'] <= 1.0
        
        logger.info(f"✓ Model evaluation: Loss={metrics['loss']:.4f}, "
                   f"Accuracy={metrics['accuracy']:.4f}")
        
        return True
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        logger.info("Testing model save/load...")
        
        config = TrainingConfig(learning_rate=0.001, local_epochs=1)
        client = FederatedLearningClient(
            client_id="test_client_004",
            server_url="http://localhost:8000",
            config=config
        )
        
        # Train model briefly
        features, labels = generate_test_features(10)
        client.train_local_model(features, labels)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            success = client.save_model(filepath)
            assert success
            
            # Create new client and load model
            new_client = FederatedLearningClient(
                client_id="test_client_005",
                server_url="http://localhost:8000",
                config=config
            )
            
            success = new_client.load_model(filepath)
            assert success
            
            # Verify loaded state
            assert new_client.current_round == client.current_round
            assert len(new_client.training_history) == len(client.training_history)
            
            logger.info("✓ Model save/load successful")
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
        
        return True


def test_integration():
    """Integration test of federated learning pipeline"""
    logger.info("Running federated learning integration test...")
    
    # Create client with comprehensive configuration
    config = TrainingConfig(
        learning_rate=0.01,
        batch_size=16,
        local_epochs=3,
        compression_method=CompressionMethod.QUANTIZATION,
        privacy_method=PrivacyMethod.DIFFERENTIAL_PRIVACY,
        privacy_epsilon=1.0,
        quantization_bits=8
    )
    
    client = FederatedLearningClient(
        client_id="integration_test_client",
        server_url="http://localhost:8000",
        config=config
    )
    
    # Generate training and test data
    train_features, train_labels = generate_test_features(50)
    test_features, test_labels = generate_test_features(20)
    
    # Initial evaluation
    initial_metrics = client.evaluate_model(test_features, test_labels)
    logger.info(f"Initial accuracy: {initial_metrics['accuracy']:.4f}")
    
    # Perform local training
    model_update = client.train_local_model(train_features, train_labels)
    
    # Final evaluation
    final_metrics = client.evaluate_model(test_features, test_labels)
    logger.info(f"Final accuracy: {final_metrics['accuracy']:.4f}")
    
    # Get training status
    status = client.get_training_status()
    
    # Verify integration
    assert model_update is not None
    assert len(model_update.model_weights) > 0
    assert model_update.training_metrics['num_samples'] == 50
    assert status['current_round'] == 0
    assert len(status['training_history']) == 1
    
    logger.info("✓ Integration test completed successfully!")
    logger.info(f"✓ Model update size: {len(model_update.model_weights)} bytes")
    logger.info(f"✓ Compression ratio: {model_update.compression_info['compression_ratio']:.3f}")
    logger.info(f"✓ Privacy cost: {model_update.privacy_info['privacy_cost']}")
    
    return True


def main():
    """Run all federated learning tests"""
    logger.info("Starting Federated Learning Tests")
    logger.info("=" * 50)
    
    test_classes = [
        ("Signal Dataset", TestSignalDataset),
        ("Signal Classifier Model", TestSignalClassifierModel),
        ("Model Compressor", TestModelCompressor),
        ("Privacy Engine", TestPrivacyEngine),
        ("Federated Learning Client", TestFederatedLearningClient)
    ]
    
    passed = 0
    total = 0
    
    for class_name, test_class in test_classes:
        logger.info(f"\n--- {class_name} Tests ---")
        test_instance = test_class()
        
        # Get test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total += 1
            try:
                logger.info(f"Running {method_name}...")
                result = getattr(test_instance, method_name)()
                if result:
                    passed += 1
                    logger.info(f"✓ {method_name} PASSED")
                else:
                    logger.error(f"✗ {method_name} FAILED")
            except Exception as e:
                logger.error(f"✗ {method_name} FAILED with exception: {e}")
    
    # Run integration test
    total += 1
    try:
        logger.info(f"\n--- Integration Test ---")
        if test_integration():
            passed += 1
            logger.info("✓ Integration test PASSED")
        else:
            logger.error("✗ Integration test FAILED")
    except Exception as e:
        logger.error(f"✗ Integration test FAILED with exception: {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All federated learning tests passed!")
        return True
    else:
        logger.warning("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)