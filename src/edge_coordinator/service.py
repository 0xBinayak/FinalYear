"""
Edge Coordinator Service - Main service integrating all edge coordinator functionality
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import threading

from .coordinator import EdgeCoordinator, CoordinatorState
from .network_partition import NetworkPartitionDetector
from .offline_operation import OfflineOperationManager, OfflineMode
from .resource_manager import ResourceManager, TrainingTask
from .data_quality import DataQualityValidator
from ..common.interfaces import ClientInfo
from ..common.federated_data_structures import EnhancedModelUpdate


class EdgeCoordinatorService:
    """
    Main Edge Coordinator Service that integrates:
    - Local client management
    - Hierarchical aggregation
    - Network partition detection
    - Offline operation with eventual consistency
    """
    
    def __init__(self, coordinator_id: str, region: str, config: Dict[str, Any]):
        self.coordinator_id = coordinator_id
        self.region = region
        self.config = config
        
        # Initialize components
        self.coordinator = EdgeCoordinator(coordinator_id, region, config)
        
        partition_config = config.get('partition_detection', {})
        self.partition_detector = NetworkPartitionDetector(coordinator_id, partition_config)
        
        storage_path = config.get('storage_path', f'./data/edge_coordinator_{coordinator_id}')
        offline_config = config.get('offline_operation', {})
        self.offline_manager = OfflineOperationManager(coordinator_id, storage_path, offline_config)
        
        # Resource management and optimization
        resource_config = config.get('resource_management', {})
        self.resource_manager = ResourceManager(coordinator_id, resource_config)
        
        # Data quality validation
        quality_config = config.get('data_quality', {})
        self.data_quality_validator = DataQualityValidator(coordinator_id, quality_config)
        
        # Service state
        self.running = False
        self.start_time: Optional[datetime] = None
        
        # Logging
        self.logger = logging.getLogger(f"EdgeCoordinatorService-{coordinator_id}")
        
        # FastAPI app for REST API
        self.app = self._create_fastapi_app()
        
        # Background coordination task
        self.coordination_task: Optional[asyncio.Task] = None
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application for edge coordinator"""
        app = FastAPI(
            title=f"Edge Coordinator {self.coordinator_id}",
            description="Edge Coordinator for Federated Learning",
            version="1.0.0"
        )
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "coordinator_id": self.coordinator_id,
                "region": self.region,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
        
        @app.get("/status")
        async def get_status():
            """Get comprehensive coordinator status"""
            return {
                "coordinator": self.coordinator.get_coordinator_status(),
                "partition_detection": self.partition_detector.get_partition_status(),
                "offline_operation": self.offline_manager.get_offline_status(),
                "resource_management": self.resource_manager.get_resource_status(),
                "data_quality": self.data_quality_validator.get_validation_status()
            }
        
        @app.post("/clients/register")
        async def register_client(client_info: dict):
            """Register a new local client"""
            try:
                # Convert dict to ClientInfo
                client_info_obj = ClientInfo(
                    client_id=client_info['client_id'],
                    client_type=client_info['client_type'],
                    capabilities=client_info.get('capabilities', {}),
                    location=client_info.get('location'),
                    network_info=client_info.get('network_info', {}),
                    hardware_specs=client_info.get('hardware_specs', {}),
                    reputation_score=client_info.get('reputation_score', 1.0)
                )
                
                token = self.coordinator.register_local_client(client_info_obj)
                
                # Register with partition detector
                self.partition_detector.register_node(
                    client_info_obj.client_id,
                    {
                        'client_type': client_info_obj.client_type,
                        'endpoint': client_info.get('endpoint'),
                        'ip_address': client_info.get('ip_address')
                    }
                )
                
                # Register with resource manager
                self.resource_manager.register_client(client_info_obj)
                
                return {"token": token, "coordinator_id": self.coordinator_id}
                
            except Exception as e:
                self.logger.error(f"Client registration failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.delete("/clients/{client_id}")
        async def unregister_client(client_id: str):
            """Unregister a local client"""
            success = self.coordinator.unregister_local_client(client_id)
            if success:
                return {"message": f"Client {client_id} unregistered"}
            else:
                raise HTTPException(status_code=404, detail="Client not found")
        
        @app.post("/clients/{client_id}/heartbeat")
        async def client_heartbeat(client_id: str, metadata: dict = None):
            """Receive client heartbeat"""
            self.partition_detector.update_node_heartbeat(client_id, metadata)
            return {"status": "acknowledged"}
        
        @app.post("/clients/{client_id}/model_update")
        async def receive_model_update(client_id: str, update_data: dict):
            """Receive model update from local client"""
            try:
                # Convert dict to EnhancedModelUpdate
                # This is a simplified conversion - in practice, you'd have proper serialization
                update = EnhancedModelUpdate(
                    client_id=client_id,
                    model_weights=bytes.fromhex(update_data['model_weights']),
                    model_size_bytes=update_data['model_size_bytes'],
                    training_rounds=update_data.get('training_rounds', 1),
                    local_epochs=update_data.get('local_epochs', 1),
                    batch_size=update_data.get('batch_size', 32),
                    learning_rate=update_data.get('learning_rate', 0.01),
                    samples_used=update_data.get('samples_used', 100),
                    training_time_seconds=update_data.get('training_time_seconds', 60.0),
                    training_loss=update_data.get('training_loss', 1.0)
                )
                
                success = self.coordinator.receive_local_update(client_id, update)
                
                if success:
                    return {"status": "accepted", "update_id": f"update-{client_id}-{int(datetime.now().timestamp())}"}
                else:
                    raise HTTPException(status_code=400, detail="Update rejected")
                
            except Exception as e:
                self.logger.error(f"Model update processing failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/aggregation/trigger")
        async def trigger_aggregation(background_tasks: BackgroundTasks):
            """Trigger local model aggregation"""
            background_tasks.add_task(self._perform_aggregation)
            return {"status": "aggregation_triggered"}
        
        @app.post("/sync/global")
        async def sync_with_global(server_endpoint: str):
            """Sync with global aggregation server"""
            try:
                if self.offline_manager.mode != OfflineMode.NORMAL:
                    # Attempt to sync offline updates
                    result = await self.offline_manager.sync_with_global_server(server_endpoint)
                    return result
                else:
                    # Normal sync
                    success = self.coordinator.sync_with_global_server(server_endpoint)
                    return {"status": "success" if success else "failed"}
                
            except Exception as e:
                self.logger.error(f"Global sync failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/partition/status")
        async def get_partition_status():
            """Get network partition status"""
            return self.partition_detector.get_partition_status()
        
        @app.get("/offline/status")
        async def get_offline_status():
            """Get offline operation status"""
            return self.offline_manager.get_offline_status()
        
        @app.post("/offline/enter")
        async def enter_offline_mode(reason: str = "manual"):
            """Manually enter offline mode"""
            self.offline_manager.enter_offline_mode(reason)
            return {"status": "offline_mode_entered", "reason": reason}
        
        @app.post("/offline/exit")
        async def exit_offline_mode():
            """Manually exit offline mode"""
            self.offline_manager.exit_offline_mode()
            return {"status": "offline_mode_exited"}
        
        # Resource Management Endpoints
        @app.get("/resources/status")
        async def get_resource_status():
            """Get resource management status"""
            return self.resource_manager.get_resource_status()
        
        @app.post("/resources/clients/{client_id}/update")
        async def update_client_resources(client_id: str, resource_update: dict):
            """Update client resource information"""
            success = self.resource_manager.update_client_resources(client_id, resource_update)
            if success:
                return {"status": "updated"}
            else:
                raise HTTPException(status_code=404, detail="Client not found")
        
        @app.get("/resources/clients/{client_id}/capability")
        async def get_client_capability(client_id: str):
            """Get client capability assessment"""
            capability = self.resource_manager.assess_device_capability(client_id)
            if 'error' in capability:
                raise HTTPException(status_code=404, detail=capability['error'])
            return capability
        
        @app.post("/resources/tasks/schedule")
        async def schedule_training_task(task_data: dict):
            """Schedule a training task"""
            try:
                task = TrainingTask(
                    task_id=task_data['task_id'],
                    client_id=task_data.get('client_id', ''),
                    model_size_mb=task_data['model_size_mb'],
                    batch_size=task_data['batch_size'],
                    estimated_epochs=task_data['estimated_epochs'],
                    estimated_duration_minutes=task_data['estimated_duration_minutes'],
                    priority=task_data.get('priority', 1.0)
                )
                
                success = self.resource_manager.schedule_training_task(task)
                if success:
                    return {"status": "scheduled", "task_id": task.task_id}
                else:
                    raise HTTPException(status_code=400, detail="Failed to schedule task")
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/resources/load_balancing")
        async def get_load_balancing_recommendations():
            """Get load balancing recommendations"""
            return self.resource_manager.get_load_balancing_recommendations()
        
        # Data Quality Endpoints
        @app.get("/quality/status")
        async def get_quality_status():
            """Get data quality validation status"""
            return self.data_quality_validator.get_validation_status()
        
        @app.get("/quality/clients/{client_id}/summary")
        async def get_client_quality_summary(client_id: str):
            """Get quality summary for a specific client"""
            summary = self.data_quality_validator.get_client_quality_summary(client_id)
            if 'error' in summary:
                raise HTTPException(status_code=404, detail=summary['error'])
            return summary
        
        return app
    
    async def start(self):
        """Start the edge coordinator service"""
        self.running = True
        self.start_time = datetime.now()
        
        # Start all components
        await self.coordinator.start()
        await self.partition_detector.start()
        await self.offline_manager.start()
        await self.resource_manager.start()
        await self.data_quality_validator.start()
        
        # Start coordination task
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        
        self.logger.info(f"Edge Coordinator Service {self.coordinator_id} started in region {self.region}")
    
    async def stop(self):
        """Stop the edge coordinator service"""
        self.running = False
        
        # Stop coordination task
        if self.coordination_task:
            self.coordination_task.cancel()
            try:
                await self.coordination_task
            except asyncio.CancelledError:
                pass
        
        # Stop all components
        await self.coordinator.stop()
        await self.partition_detector.stop()
        await self.offline_manager.stop()
        await self.resource_manager.stop()
        await self.data_quality_validator.stop()
        
        self.logger.info(f"Edge Coordinator Service {self.coordinator_id} stopped")
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while self.running:
            try:
                await self._coordinate_operations()
                await asyncio.sleep(30)  # Coordinate every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(30)
    
    async def _coordinate_operations(self):
        """Coordinate between different components"""
        # Check partition status and adjust operation mode
        partition_status = self.partition_detector.get_partition_status()
        
        # If we have active partitions, consider entering offline mode
        if partition_status['active_partitions'] > 0:
            global_partitions = [
                p for p in self.partition_detector.active_partitions.values()
                if 'global' in p.partition_type.value
            ]
            
            if global_partitions and self.offline_manager.mode == OfflineMode.NORMAL:
                self.offline_manager.enter_offline_mode("global_partition_detected")
        
        # If no partitions and we're offline, try to exit offline mode
        elif (partition_status['active_partitions'] == 0 and 
              self.offline_manager.mode in [OfflineMode.ISOLATED, OfflineMode.DEGRADED]):
            self.offline_manager.exit_offline_mode()
        
        # Perform periodic aggregation if we have pending updates
        if len(self.coordinator.pending_updates) >= 3:  # Aggregate when we have 3+ updates
            await self._perform_aggregation()
    
    async def _perform_aggregation(self):
        """Perform local model aggregation"""
        try:
            aggregated_update = await self.coordinator.aggregate_local_models()
            
            if aggregated_update:
                # If we're offline, store the aggregated update
                if self.offline_manager.mode != OfflineMode.NORMAL:
                    update_id = self.offline_manager.store_offline_update(
                        aggregated_update, 
                        priority=1.5  # Higher priority for aggregated updates
                    )
                    self.logger.info(f"Stored aggregated update offline: {update_id}")
                else:
                    # Try to send to global server immediately
                    # This would be implemented with actual HTTP/gRPC calls
                    self.logger.info("Aggregated update ready for global sync")
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the FastAPI server"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Run server in background thread
        def run_in_thread():
            asyncio.run(server.serve())
        
        server_thread = threading.Thread(target=run_in_thread, daemon=True)
        server_thread.start()
        
        self.logger.info(f"Edge Coordinator API server started on {host}:{port}")
        return server_thread


async def create_edge_coordinator_service(coordinator_id: str, region: str, 
                                        config_path: Optional[str] = None) -> EdgeCoordinatorService:
    """Factory function to create and configure edge coordinator service"""
    
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'max_local_clients': 20,
            'heartbeat_interval': 30,
            'sync_interval': 300,
            'aggregation_strategy': 'fedavg',
            'partition_detection': {
                'heartbeat_timeout': 60,
                'probe_interval': 30,
                'consensus_threshold': 0.6,
                'global_servers': ['http://global-server:8000'],
                'peer_coordinators': []
            },
            'offline_operation': {
                'max_offline_updates': 1000,
                'max_offline_hours': 24,
                'sync_strategy': 'batched',
                'compression_enabled': True
            },
            'resource_management': {
                'scheduling_strategy': 'resource_aware',
                'max_concurrent_training': 5,
                'resource_update_interval': 60,
                'load_balancing_enabled': True
            },
            'data_quality': {
                'min_snr_db': -10.0,
                'max_noise_floor_db': -80.0,
                'min_samples_per_class': 10,
                'max_frequency_drift_hz': 1000.0,
                'outlier_threshold_std': 3.0,
                'auto_preprocessing': True,
                'noise_reduction_enabled': True,
                'normalization_enabled': True,
                'outlier_removal_enabled': True
            },
            'storage_path': f'./data/edge_coordinator_{coordinator_id}'
        }
    
    # Create service
    service = EdgeCoordinatorService(coordinator_id, region, config)
    
    return service


# CLI entry point for running edge coordinator
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Edge Coordinator Service")
    parser.add_argument("--coordinator-id", required=True, help="Coordinator ID")
    parser.add_argument("--region", required=True, help="Region name")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    
    args = parser.parse_args()
    
    async def main():
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Create and start service
            service = await create_edge_coordinator_service(
                args.coordinator_id, 
                args.region, 
                args.config
            )
            
            await service.start()
            
            # Start API server
            server_thread = service.run_server(args.host, args.port)
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
            
            # Cleanup
            await service.stop()
            
        except Exception as e:
            print(f"Failed to start edge coordinator: {e}")
            sys.exit(1)
    
    asyncio.run(main())