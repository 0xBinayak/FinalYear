"""
Model storage and versioning system
"""
import os
import json
import pickle
import hashlib
import asyncio
import aiofiles
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from ..common.config import AppConfig

logger = logging.getLogger(__name__)


class ModelStorage:
    """Model storage and versioning system"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.storage_dir = Path("models")
        self.metadata_file = self.storage_dir / "metadata.json"
        self.models_dir = self.storage_dir / "versions"
        self.checkpoints_dir = self.storage_dir / "checkpoints"
        
        # In-memory cache
        self.model_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, Any] = {}
        
        # Storage configuration
        self.max_versions = 50  # Keep last 50 versions
        self.compression_enabled = True
        self.backup_enabled = True
    
    async def initialize(self):
        """Initialize storage system"""
        logger.info("Initializing model storage...")
        
        # Create directories
        self.storage_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        await self._load_metadata()
        
        # Cleanup old versions if needed
        await self._cleanup_old_versions()
        
        logger.info("Model storage initialized successfully")
    
    async def shutdown(self):
        """Shutdown storage system"""
        logger.info("Shutting down model storage...")
        
        # Save metadata
        await self._save_metadata()
        
        # Clear cache
        self.model_cache.clear()
        self.metadata_cache.clear()
        
        logger.info("Model storage shutdown complete")
    
    async def store_model(self, model_data: Dict[str, Any]) -> str:
        """Store model with versioning"""
        try:
            # Generate model ID
            model_id = self._generate_model_id(model_data)
            
            # Prepare model metadata
            metadata = {
                'id': model_id,
                'version': model_data.get('version', '1.0.0'),
                'round': model_data.get('round', 0),
                'timestamp': model_data.get('timestamp', datetime.utcnow()).isoformat(),
                'participating_clients': model_data.get('participating_clients', []),
                'size_bytes': len(model_data['weights']),
                'checksum': self._calculate_checksum(model_data['weights'])
            }
            
            # Store model weights
            model_path = self.models_dir / f"{model_id}.pkl"
            async with aiofiles.open(model_path, 'wb') as f:
                await f.write(model_data['weights'])
            
            # Store metadata
            metadata_path = self.models_dir / f"{model_id}.json"
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            # Update global metadata
            self.metadata_cache[model_id] = metadata
            await self._save_metadata()
            
            # Cache model
            self.model_cache[model_id] = model_data
            
            logger.info(f"Model {model_id} stored successfully")
            return model_id
        
        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            raise
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID"""
        try:
            # Check cache first
            if model_id in self.model_cache:
                return self.model_cache[model_id]
            
            # Load from disk
            model_path = self.models_dir / f"{model_id}.pkl"
            metadata_path = self.models_dir / f"{model_id}.json"
            
            if not model_path.exists() or not metadata_path.exists():
                return None
            
            # Load metadata
            async with aiofiles.open(metadata_path, 'r') as f:
                metadata = json.loads(await f.read())
            
            # Load model weights
            async with aiofiles.open(model_path, 'rb') as f:
                weights = await f.read()
            
            model_data = {
                'weights': weights,
                'metadata': metadata,
                'version': metadata['version'],
                'round': metadata['round'],
                'timestamp': datetime.fromisoformat(metadata['timestamp'])
            }
            
            # Cache model
            self.model_cache[model_id] = model_data
            
            return model_data
        
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    async def get_latest_model(self) -> Optional[Dict[str, Any]]:
        """Get the latest model"""
        if not self.metadata_cache:
            return None
        
        # Find latest model by timestamp
        latest_id = max(
            self.metadata_cache.keys(),
            key=lambda x: self.metadata_cache[x]['timestamp']
        )
        
        return await self.get_model(latest_id)
    
    async def get_model_by_round(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Get model by training round"""
        for model_id, metadata in self.metadata_cache.items():
            if metadata['round'] == round_number:
                return await self.get_model(model_id)
        return None
    
    async def list_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List stored models"""
        # Sort by timestamp (newest first)
        sorted_models = sorted(
            self.metadata_cache.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        return [metadata for _, metadata in sorted_models[:limit]]
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model by ID"""
        try:
            # Remove files
            model_path = self.models_dir / f"{model_id}.pkl"
            metadata_path = self.models_dir / f"{model_id}.json"
            
            if model_path.exists():
                model_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from cache and metadata
            self.model_cache.pop(model_id, None)
            self.metadata_cache.pop(model_id, None)
            
            await self._save_metadata()
            
            logger.info(f"Model {model_id} deleted successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    async def create_checkpoint(self, model_id: str, checkpoint_name: str) -> bool:
        """Create a checkpoint of a model"""
        try:
            model_data = await self.get_model(model_id)
            if not model_data:
                return False
            
            # Create checkpoint directory
            checkpoint_dir = self.checkpoints_dir / checkpoint_name
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Copy model files
            model_path = self.models_dir / f"{model_id}.pkl"
            metadata_path = self.models_dir / f"{model_id}.json"
            
            checkpoint_model_path = checkpoint_dir / f"{model_id}.pkl"
            checkpoint_metadata_path = checkpoint_dir / f"{model_id}.json"
            
            # Copy files
            async with aiofiles.open(model_path, 'rb') as src:
                async with aiofiles.open(checkpoint_model_path, 'wb') as dst:
                    await dst.write(await src.read())
            
            async with aiofiles.open(metadata_path, 'r') as src:
                async with aiofiles.open(checkpoint_metadata_path, 'w') as dst:
                    await dst.write(await src.read())
            
            logger.info(f"Checkpoint '{checkpoint_name}' created for model {model_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return False
    
    async def restore_checkpoint(self, checkpoint_name: str) -> Optional[str]:
        """Restore model from checkpoint"""
        try:
            checkpoint_dir = self.checkpoints_dir / checkpoint_name
            if not checkpoint_dir.exists():
                return None
            
            # Find model files in checkpoint
            model_files = list(checkpoint_dir.glob("*.pkl"))
            if not model_files:
                return None
            
            model_file = model_files[0]
            model_id = model_file.stem
            metadata_file = checkpoint_dir / f"{model_id}.json"
            
            # Load checkpoint data
            async with aiofiles.open(model_file, 'rb') as f:
                weights = await f.read()
            
            async with aiofiles.open(metadata_file, 'r') as f:
                metadata = json.loads(await f.read())
            
            # Create new model with restored data
            model_data = {
                'weights': weights,
                'version': metadata['version'],
                'round': metadata['round'],
                'timestamp': datetime.utcnow(),
                'participating_clients': metadata['participating_clients']
            }
            
            new_model_id = await self.store_model(model_data)
            
            logger.info(f"Model restored from checkpoint '{checkpoint_name}' as {new_model_id}")
            return new_model_id
        
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_models = len(self.metadata_cache)
            total_size = sum(
                metadata['size_bytes'] 
                for metadata in self.metadata_cache.values()
            )
            
            # Calculate disk usage
            disk_usage = sum(
                f.stat().st_size 
                for f in self.storage_dir.rglob('*') 
                if f.is_file()
            )
            
            return {
                'total_models': total_models,
                'total_size_bytes': total_size,
                'disk_usage_bytes': disk_usage,
                'cache_size': len(self.model_cache),
                'storage_directory': str(self.storage_dir),
                'oldest_model': min(
                    self.metadata_cache.values(),
                    key=lambda x: x['timestamp']
                )['timestamp'] if self.metadata_cache else None,
                'newest_model': max(
                    self.metadata_cache.values(),
                    key=lambda x: x['timestamp']
                )['timestamp'] if self.metadata_cache else None
            }
        
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def _generate_model_id(self, model_data: Dict[str, Any]) -> str:
        """Generate unique model ID"""
        # Create ID based on content and timestamp
        content = f"{model_data.get('round', 0)}_{model_data.get('version', '1.0.0')}_{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data integrity"""
        return hashlib.md5(data).hexdigest()
    
    async def _load_metadata(self):
        """Load metadata from disk"""
        try:
            if self.metadata_file.exists():
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    self.metadata_cache = json.loads(await f.read())
                logger.info(f"Loaded metadata for {len(self.metadata_cache)} models")
            else:
                self.metadata_cache = {}
        
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata_cache = {}
    
    async def _save_metadata(self):
        """Save metadata to disk"""
        try:
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(self.metadata_cache, indent=2))
        
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def _cleanup_old_versions(self):
        """Cleanup old model versions"""
        if len(self.metadata_cache) <= self.max_versions:
            return
        
        # Sort by timestamp and keep only the latest versions
        sorted_models = sorted(
            self.metadata_cache.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        models_to_delete = sorted_models[:-self.max_versions]
        
        for model_id, _ in models_to_delete:
            await self.delete_model(model_id)
        
        logger.info(f"Cleaned up {len(models_to_delete)} old model versions")