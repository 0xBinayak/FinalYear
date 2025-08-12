"""
Test suite for mobile optimizations
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch
import numpy as np

from .mobile_optimizations import (
    BackgroundTrainingManager, BackgroundTrainingConfig, BackgroundTrainingState,
    NetworkHandoffManager, NetworkHandoffState,
    AdaptiveModelComplexityManager, ModelComplexityProfile,
    IncentiveReputationManager, IncentiveMetrics
)
from ..common.federated_data_structures import NetworkConditions, ComputeResources


class TestBackgroundTrainingManager:
    """Test background training management"""
    
    def test_background_training_config(self):
        """Test background training configuration"""
        config = BackgroundTrainingConfig(
            enabled=True,
            max_background_time_minutes=30,
            min_idle_time_seconds=180
        )
        
        assert config.enabled is True
        assert config.max_background_time_minutes == 30
        assert config.min_idle_time_seconds == 180
    
    def test_background_training_manager_creation(self):
        """Test creating background training manager"""
        config = BackgroundTrainingConfig()
        manager = BackgroundTrainingManager(config)
        
        assert manager.state == BackgroundTrainingState.IDLE
        assert not manager.system_monitor_active
        assert manager.training_callback is None
    
    def test_training_callback_setting(self):
        """Test setting training callback"""
        config = BackgroundTrainingConfig()
        manager = BackgroundTrainingManager(config)
        
        def mock_callback(params):
            return {'success': True}
        
        manager.set_training_callback(mock_callback)
        assert manager.training_callback == mock_callback
    
    def test_system_monitoring_lifecycle(self):
        """Test system monitoring start/stop"""
        config = BackgroundTrainingConfig()
        manager = BackgroundTrainingManager(config)
        
        # Start monitoring
        manager.start_system_monitoring()
        assert manager.system_monitor_active is True
        
        # Stop monitoring
        manager.stop_system_monitoring()
        assert manager.system_monitor_active is False
    
    def test_training_state_transitions(self):
        """Test training state transitions"""
        config = BackgroundTrainingConfig()
        manager = BackgroundTrainingManager(config)
        
        # Initial state
        assert manager.state == BackgroundTrainingState.IDLE
        
        # Pause (should not change from idle)
        manager.pause_training()
        assert manager.state == BackgroundTrainingState.IDLE
        
        # Simulate active state
        manager.state = BackgroundTrainingState.ACTIVE
        manager.pause_training()
        assert manager.state == BackgroundTrainingState.PAUSED
        
        # Resume
        manager.resume_training()
        assert manager.state == BackgroundTrainingState.ACTIVE
        
        # Suspend
        manager.suspend_training()
        assert manager.state == BackgroundTrainingState.SUSPENDED
        
        # Resume from suspension
        manager.resume_from_suspension()
        assert manager.state == BackgroundTrainingState.IDLE
    
    def test_user_activity_notification(self):
        """Test user activity notification"""
        config = BackgroundTrainingConfig(pause_on_user_activity=True)
        manager = BackgroundTrainingManager(config)
        
        initial_time = manager.last_activity_time
        time.sleep(0.1)
        
        manager.notify_user_activity()
        assert manager.last_activity_time > initial_time
    
    def test_training_statistics(self):
        """Test training statistics"""
        config = BackgroundTrainingConfig()
        manager = BackgroundTrainingManager(config)
        
        stats = manager.get_training_statistics()
        
        assert 'total_sessions' in stats
        assert 'total_background_time_minutes' in stats
        assert 'current_state' in stats
        assert stats['current_state'] == BackgroundTrainingState.IDLE.value


class TestNetworkHandoffManager:
    """Test network handoff management"""
    
    def test_network_handoff_manager_creation(self):
        """Test creating network handoff manager"""
        manager = NetworkHandoffManager()
        
        assert manager.current_state == NetworkHandoffState.STABLE
        assert manager.previous_network is None
        assert manager.current_network is None
        assert not manager.monitoring_active
    
    def test_callback_setting(self):
        """Test setting callbacks"""
        manager = NetworkHandoffManager()
        
        def mock_handoff_callback(event):
            pass
        
        def mock_reconnection_callback(event):
            pass
        
        manager.set_handoff_callback(mock_handoff_callback)
        manager.set_reconnection_callback(mock_reconnection_callback)
        
        assert manager.handoff_callback == mock_handoff_callback
        assert manager.reconnection_callback == mock_reconnection_callback
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop"""
        manager = NetworkHandoffManager()
        
        # Start monitoring
        manager.start_monitoring()
        assert manager.monitoring_active is True
        
        # Stop monitoring
        manager.stop_monitoring()
        assert manager.monitoring_active is False
    
    def test_network_info_generation(self):
        """Test network info generation"""
        manager = NetworkHandoffManager()
        
        network_info = manager._get_current_network_info()
        
        assert 'type' in network_info
        assert 'bandwidth_mbps' in network_info
        assert 'latency_ms' in network_info
        assert 'stability' in network_info
        assert 'is_metered' in network_info
        assert 'interface_id' in network_info
    
    def test_network_change_detection(self):
        """Test network change detection"""
        manager = NetworkHandoffManager()
        
        # First network
        network1 = {
            'interface_id': 'wifi_1',
            'bandwidth_mbps': 50.0,
            'latency_ms': 20.0
        }
        
        # No previous network - should detect change
        assert manager._detect_network_change(network1) is True
        
        manager.current_network = network1
        
        # Same network - should not detect change
        assert manager._detect_network_change(network1) is False
        
        # Different interface - should detect change
        network2 = {
            'interface_id': 'cellular_1',
            'bandwidth_mbps': 50.0,
            'latency_ms': 20.0
        }
        assert manager._detect_network_change(network2) is True
        
        # Significant bandwidth change - should detect change
        network3 = {
            'interface_id': 'wifi_1',
            'bandwidth_mbps': 10.0,  # 40 Mbps difference
            'latency_ms': 20.0
        }
        assert manager._detect_network_change(network3) is True
    
    def test_network_status(self):
        """Test network status reporting"""
        manager = NetworkHandoffManager()
        
        status = manager.get_network_status()
        
        assert 'current_state' in status
        assert 'current_network' in status
        assert 'previous_network' in status
        assert 'handoff_events_count' in status
        assert 'is_stable' in status
        assert status['current_state'] == NetworkHandoffState.STABLE.value


class TestAdaptiveModelComplexityManager:
    """Test adaptive model complexity management"""
    
    def test_complexity_manager_creation(self):
        """Test creating complexity manager"""
        manager = AdaptiveModelComplexityManager()
        
        assert len(manager.complexity_profiles) > 0
        assert 'minimal' in manager.complexity_profiles
        assert 'low' in manager.complexity_profiles
        assert 'medium' in manager.complexity_profiles
        assert 'high' in manager.complexity_profiles
        assert 'maximum' in manager.complexity_profiles
        assert manager.current_profile is None
    
    def test_complexity_profiles(self):
        """Test complexity profiles"""
        manager = AdaptiveModelComplexityManager()
        
        minimal_profile = manager.complexity_profiles['minimal']
        maximum_profile = manager.complexity_profiles['maximum']
        
        # Minimal should have fewer parameters than maximum
        assert minimal_profile.max_parameters < maximum_profile.max_parameters
        assert minimal_profile.max_layers < maximum_profile.max_layers
        assert minimal_profile.memory_requirement_mb < maximum_profile.memory_requirement_mb
        assert minimal_profile.training_time_estimate_minutes < maximum_profile.training_time_estimate_minutes
    
    def test_profile_selection_high_resources(self):
        """Test profile selection with high resources"""
        manager = AdaptiveModelComplexityManager()
        
        # High resource scenario
        compute_resources = ComputeResources(
            cpu_cores=8,
            cpu_frequency_ghz=3.0,
            memory_gb=16.0,
            available_memory_gb=12.0,
            battery_level=0.8,
            thermal_state="normal",
            power_source="plugged"
        )
        
        network_conditions = NetworkConditions(
            bandwidth_mbps=100.0,
            latency_ms=10.0,
            packet_loss_rate=0.001,
            jitter_ms=2.0,
            connection_stability=0.95,
            is_metered=False
        )
        
        selected_profile = manager.select_optimal_profile(compute_resources, network_conditions)
        
        # Should select a high complexity profile
        assert selected_profile.profile_name in ['high', 'maximum']
        assert manager.current_profile == selected_profile
    
    def test_profile_selection_low_resources(self):
        """Test profile selection with low resources"""
        manager = AdaptiveModelComplexityManager()
        
        # Low resource scenario
        compute_resources = ComputeResources(
            cpu_cores=2,
            cpu_frequency_ghz=1.5,
            memory_gb=2.0,
            available_memory_gb=0.8,
            battery_level=0.2,
            thermal_state="hot",
            power_source="battery"
        )
        
        network_conditions = NetworkConditions(
            bandwidth_mbps=2.0,
            latency_ms=200.0,
            packet_loss_rate=0.05,
            jitter_ms=50.0,
            connection_stability=0.6,
            is_metered=True
        )
        
        selected_profile = manager.select_optimal_profile(compute_resources, network_conditions)
        
        # Should select a low complexity profile
        assert selected_profile.profile_name in ['minimal', 'low']
    
    def test_custom_profile_addition(self):
        """Test adding custom profile"""
        manager = AdaptiveModelComplexityManager()
        
        custom_profile = ModelComplexityProfile(
            profile_name='custom_test',
            max_parameters=75000,
            max_layers=6,
            batch_size=24,
            learning_rate=0.003,
            epochs=6,
            memory_requirement_mb=150,
            compute_requirement_flops=8e6,
            training_time_estimate_minutes=8
        )
        
        manager.add_custom_profile(custom_profile)
        
        assert 'custom_test' in manager.complexity_profiles
        assert manager.complexity_profiles['custom_test'] == custom_profile
    
    def test_adaptation_history(self):
        """Test adaptation history tracking"""
        manager = AdaptiveModelComplexityManager()
        
        compute_resources = ComputeResources(
            cpu_cores=4, cpu_frequency_ghz=2.0, memory_gb=8.0, available_memory_gb=6.0
        )
        network_conditions = NetworkConditions(
            bandwidth_mbps=50.0, latency_ms=30.0, packet_loss_rate=0.01,
            jitter_ms=10.0, connection_stability=0.8
        )
        
        # Make selection
        manager.select_optimal_profile(compute_resources, network_conditions)
        
        history = manager.get_adaptation_history()
        assert len(history) == 1
        assert 'timestamp' in history[0]
        assert 'selected_profile' in history[0]
        assert 'profile_scores' in history[0]


class TestIncentiveReputationManager:
    """Test incentive and reputation management"""
    
    def test_incentive_manager_creation(self):
        """Test creating incentive manager"""
        manager = IncentiveReputationManager("test_client")
        
        assert manager.client_id == "test_client"
        assert isinstance(manager.metrics, IncentiveMetrics)
        assert manager.metrics.participation_count == 0
        assert manager.metrics.reputation_score == 1.0
    
    def test_successful_training_participation(self):
        """Test recording successful training participation"""
        manager = IncentiveReputationManager("test_client")
        
        training_result = {
            'success': True,
            'samples_used': 100,
            'training_time_minutes': 10,
            'model_quality_score': 0.8,
            'efficiency_score': 0.7,
            'data_quality_score': 0.85
        }
        
        record = manager.record_training_participation(training_result)
        
        assert manager.metrics.participation_count == 1
        assert manager.metrics.successful_trainings == 1
        assert manager.metrics.current_streak == 1
        assert manager.metrics.total_samples_contributed == 100
        assert record['rewards_earned'] > 0
        assert record['penalties_applied'] == 0
    
    def test_failed_training_participation(self):
        """Test recording failed training participation"""
        manager = IncentiveReputationManager("test_client")
        
        training_result = {
            'success': False,
            'samples_used': 50,
            'training_time_minutes': 5,
            'model_quality_score': 0.2,
            'efficiency_score': 0.3
        }
        
        record = manager.record_training_participation(training_result)
        
        assert manager.metrics.participation_count == 1
        assert manager.metrics.successful_trainings == 0
        assert manager.metrics.failed_trainings == 1
        assert manager.metrics.current_streak == 0
        assert record['penalties_applied'] > 0
    
    def test_streak_tracking(self):
        """Test streak tracking"""
        manager = IncentiveReputationManager("test_client")
        
        # Record multiple successful trainings
        for i in range(5):
            training_result = {
                'success': True,
                'samples_used': 50,
                'training_time_minutes': 5,
                'model_quality_score': 0.7,
                'efficiency_score': 0.6
            }
            manager.record_training_participation(training_result)
        
        assert manager.metrics.current_streak == 5
        assert manager.metrics.best_streak == 5
        
        # Record a failure
        training_result = {
            'success': False,
            'samples_used': 30,
            'training_time_minutes': 3
        }
        manager.record_training_participation(training_result)
        
        assert manager.metrics.current_streak == 0
        assert manager.metrics.best_streak == 5  # Best streak preserved
    
    def test_reputation_calculation(self):
        """Test reputation score calculation"""
        manager = IncentiveReputationManager("test_client")
        
        # Record high-quality training
        training_result = {
            'success': True,
            'samples_used': 100,
            'training_time_minutes': 10,
            'model_quality_score': 0.9,
            'efficiency_score': 0.8,
            'data_quality_score': 0.85,
            'network_issues': 0,
            'battery_usage_percent': 8
        }
        
        manager.record_training_participation(training_result)
        
        reputation_breakdown = manager.get_reputation_breakdown()
        
        assert 'overall_reputation' in reputation_breakdown
        assert 'success_rate' in reputation_breakdown
        assert 'quality_score' in reputation_breakdown
        assert reputation_breakdown['success_rate'] == 1.0
        assert reputation_breakdown['overall_reputation'] > 0.8
    
    def test_reward_calculation(self):
        """Test reward calculation"""
        manager = IncentiveReputationManager("test_client")
        
        # High quality training should get good rewards
        training_result = {
            'success': True,
            'model_quality_score': 0.9,
            'efficiency_score': 0.8,
            'data_quality_score': 0.9
        }
        
        rewards = manager._calculate_rewards(training_result)
        assert rewards > 1.0  # Base + bonuses
        
        # Low quality training should get minimal rewards
        training_result = {
            'success': True,
            'model_quality_score': 0.3,
            'efficiency_score': 0.2,
            'data_quality_score': 0.4
        }
        
        rewards = manager._calculate_rewards(training_result)
        assert rewards == 1.0  # Just base reward
    
    def test_penalty_calculation(self):
        """Test penalty calculation"""
        manager = IncentiveReputationManager("test_client")
        
        # Failed training with poor quality
        training_result = {
            'success': False,
            'model_quality_score': 0.1,
            'efficiency_score': 0.2,
            'network_issues': 5
        }
        
        penalties = manager._calculate_penalties(training_result)
        assert penalties > 0.5  # Multiple penalties
    
    def test_metrics_export(self):
        """Test metrics export"""
        manager = IncentiveReputationManager("test_client")
        
        # Record some activity
        training_result = {
            'success': True,
            'samples_used': 50,
            'training_time_minutes': 5,
            'model_quality_score': 0.7
        }
        manager.record_training_participation(training_result)
        
        exported_metrics = manager.export_metrics()
        
        assert 'client_id' in exported_metrics
        assert 'metrics' in exported_metrics
        assert 'reputation_breakdown' in exported_metrics
        assert 'reward_summary' in exported_metrics
        assert exported_metrics['client_id'] == "test_client"
    
    def test_reward_summary(self):
        """Test reward summary"""
        manager = IncentiveReputationManager("test_client")
        
        # Record some training
        training_result = {
            'success': True,
            'samples_used': 100,
            'model_quality_score': 0.8
        }
        manager.record_training_participation(training_result)
        
        summary = manager.get_reward_summary()
        
        assert 'total_rewards' in summary
        assert 'total_penalties' in summary
        assert 'net_rewards' in summary
        assert 'participation_count' in summary
        assert summary['participation_count'] == 1


# Integration tests
class TestMobileOptimizationsIntegration:
    """Integration tests for mobile optimizations"""
    
    def test_background_training_with_callback(self):
        """Test background training with actual callback"""
        config = BackgroundTrainingConfig(enabled=True)
        manager = BackgroundTrainingManager(config)
        
        training_executed = False
        training_result = None
        
        def training_callback(params):
            nonlocal training_executed, training_result
            training_executed = True
            training_result = params
            return {'success': True, 'model_quality': 0.8}
        
        manager.set_training_callback(training_callback)
        
        # Manually trigger training
        manager._schedule_background_training()
        
        # Wait for training to complete
        time.sleep(1.0)
        
        assert training_executed is True
        assert training_result is not None
        assert training_result['background_mode'] is True
    
    def test_network_handoff_with_callback(self):
        """Test network handoff with callback"""
        manager = NetworkHandoffManager()
        
        handoff_events = []
        
        def handoff_callback(event):
            handoff_events.append(event)
        
        manager.set_handoff_callback(handoff_callback)
        
        # Simulate network change
        network1 = {'interface_id': 'wifi_1', 'type': 'wifi', 'bandwidth_mbps': 50.0, 'latency_ms': 20.0, 'stability': 0.9}
        network2 = {'interface_id': 'cellular_1', 'type': 'cellular', 'bandwidth_mbps': 20.0, 'latency_ms': 50.0, 'stability': 0.8}
        
        manager.current_network = network1
        manager._handle_network_change(network2)
        
        # Should have recorded handoff event
        assert len(handoff_events) > 0 or len(manager.handoff_events) > 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])