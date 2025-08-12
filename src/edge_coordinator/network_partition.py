"""
Network Partition Detection and Handling for Edge Coordinator
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import socket
import subprocess
import platform

from ..common.federated_data_structures import NetworkConditions


class PartitionType(Enum):
    """Types of network partitions"""
    CLIENT_ISOLATION = "client_isolation"  # Individual clients isolated
    CLUSTER_SPLIT = "cluster_split"        # Edge cluster split
    GLOBAL_DISCONNECT = "global_disconnect"  # Lost connection to global server
    REGIONAL_PARTITION = "regional_partition"  # Regional network issues


class RecoveryStrategy(Enum):
    """Partition recovery strategies"""
    WAIT_AND_RETRY = "wait_and_retry"
    AGGRESSIVE_RECONNECT = "aggressive_reconnect"
    OFFLINE_OPERATION = "offline_operation"
    FAILOVER_COORDINATOR = "failover_coordinator"


@dataclass
class PartitionEvent:
    """Network partition event"""
    event_id: str
    partition_type: PartitionType
    detected_at: datetime
    affected_nodes: List[str]
    severity: float  # 0.0 to 1.0
    estimated_duration: Optional[timedelta] = None
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.WAIT_AND_RETRY
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectivityTest:
    """Connectivity test result"""
    target: str
    test_type: str  # ping, tcp, http
    success: bool
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class NetworkPartitionDetector:
    """
    Advanced network partition detection using multiple methods:
    - Heartbeat monitoring
    - Connectivity probes
    - Consensus-based detection
    - Network topology analysis
    """
    
    def __init__(self, coordinator_id: str, config: Dict[str, Any]):
        self.coordinator_id = coordinator_id
        self.config = config
        self.logger = logging.getLogger(f"PartitionDetector-{coordinator_id}")
        
        # Detection parameters
        self.heartbeat_timeout = config.get('heartbeat_timeout', 60)  # seconds
        self.probe_interval = config.get('probe_interval', 30)  # seconds
        self.consensus_threshold = config.get('consensus_threshold', 0.6)  # 60% agreement
        
        # State tracking
        self.node_states: Dict[str, Dict[str, Any]] = {}
        self.connectivity_history: Dict[str, List[ConnectivityTest]] = {}
        self.active_partitions: Dict[str, PartitionEvent] = {}
        
        # Network topology
        self.known_nodes: Set[str] = set()
        self.global_server_endpoints: List[str] = config.get('global_servers', [])
        self.peer_coordinators: List[str] = config.get('peer_coordinators', [])
        
        # Background tasks
        self.running = False
        self.detection_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start partition detection"""
        self.running = True
        
        self.detection_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._connectivity_prober()),
            asyncio.create_task(self._consensus_detector()),
            asyncio.create_task(self._topology_analyzer())
        ]
        
        self.logger.info("Network partition detection started")
    
    async def stop(self):
        """Stop partition detection"""
        self.running = False
        
        for task in self.detection_tasks:
            task.cancel()
        
        await asyncio.gather(*self.detection_tasks, return_exceptions=True)
        self.logger.info("Network partition detection stopped")
    
    def register_node(self, node_id: str, node_info: Dict[str, Any]):
        """Register a node for monitoring"""
        self.known_nodes.add(node_id)
        self.node_states[node_id] = {
            'info': node_info,
            'last_seen': datetime.now(),
            'status': 'active',
            'consecutive_failures': 0
        }
        self.connectivity_history[node_id] = []
        
        self.logger.info(f"Registered node {node_id} for partition detection")
    
    def update_node_heartbeat(self, node_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Update node heartbeat"""
        if node_id in self.node_states:
            self.node_states[node_id]['last_seen'] = datetime.now()
            self.node_states[node_id]['status'] = 'active'
            self.node_states[node_id]['consecutive_failures'] = 0
            
            if metadata:
                self.node_states[node_id]['metadata'] = metadata
    
    async def _heartbeat_monitor(self):
        """Monitor node heartbeats for partition detection"""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.heartbeat_timeout)
                
                for node_id, state in self.node_states.items():
                    if state['last_seen'] < timeout_threshold:
                        if state['status'] == 'active':
                            state['status'] = 'suspected'
                            state['consecutive_failures'] += 1
                            
                            self.logger.warning(f"Node {node_id} suspected of partition (heartbeat timeout)")
                            
                            # Trigger additional connectivity tests
                            await self._test_node_connectivity(node_id)
                        
                        # If node has been unresponsive for too long, mark as partitioned
                        if state['consecutive_failures'] >= 3:
                            await self._handle_node_partition(node_id, PartitionType.CLIENT_ISOLATION)
                
                await asyncio.sleep(self.heartbeat_timeout / 3)
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(self.heartbeat_timeout / 3)
    
    async def _connectivity_prober(self):
        """Actively probe network connectivity"""
        while self.running:
            try:
                # Test global server connectivity
                for server in self.global_server_endpoints:
                    await self._probe_endpoint(server, 'global_server')
                
                # Test peer coordinator connectivity
                for peer in self.peer_coordinators:
                    await self._probe_endpoint(peer, 'peer_coordinator')
                
                # Test random sample of client nodes
                client_nodes = [node for node in self.known_nodes if node.startswith('client-')]
                if client_nodes:
                    import random
                    sample_size = min(5, len(client_nodes))
                    sample_nodes = random.sample(client_nodes, sample_size)
                    
                    for node in sample_nodes:
                        await self._test_node_connectivity(node)
                
                await asyncio.sleep(self.probe_interval)
                
            except Exception as e:
                self.logger.error(f"Connectivity prober error: {e}")
                await asyncio.sleep(self.probe_interval)
    
    async def _probe_endpoint(self, endpoint: str, endpoint_type: str):
        """Probe a specific network endpoint"""
        try:
            # Parse endpoint
            if '://' in endpoint:
                # HTTP endpoint
                test_result = await self._http_probe(endpoint)
            elif ':' in endpoint:
                # TCP endpoint
                host, port = endpoint.split(':')
                test_result = await self._tcp_probe(host, int(port))
            else:
                # Assume hostname for ping
                test_result = await self._ping_probe(endpoint)
            
            # Store result
            if endpoint not in self.connectivity_history:
                self.connectivity_history[endpoint] = []
            
            self.connectivity_history[endpoint].append(test_result)
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.connectivity_history[endpoint] = [
                test for test in self.connectivity_history[endpoint]
                if test.timestamp > cutoff_time
            ]
            
            # Analyze connectivity pattern
            await self._analyze_connectivity_pattern(endpoint, endpoint_type)
            
        except Exception as e:
            self.logger.error(f"Failed to probe endpoint {endpoint}: {e}")
    
    async def _ping_probe(self, host: str) -> ConnectivityTest:
        """Perform ping connectivity test"""
        try:
            # Use system ping command
            if platform.system().lower() == 'windows':
                cmd = ['ping', '-n', '1', '-w', '5000', host]
            else:
                cmd = ['ping', '-c', '1', '-W', '5', host]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            latency_ms = (time.time() - start_time) * 1000
            
            success = result.returncode == 0
            
            return ConnectivityTest(
                target=host,
                test_type='ping',
                success=success,
                latency_ms=latency_ms if success else None,
                error_message=result.stderr if not success else None
            )
            
        except Exception as e:
            return ConnectivityTest(
                target=host,
                test_type='ping',
                success=False,
                error_message=str(e)
            )
    
    async def _tcp_probe(self, host: str, port: int) -> ConnectivityTest:
        """Perform TCP connectivity test"""
        try:
            start_time = time.time()
            
            # Create socket connection with timeout
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            
            result = sock.connect_ex((host, port))
            latency_ms = (time.time() - start_time) * 1000
            
            sock.close()
            
            success = result == 0
            
            return ConnectivityTest(
                target=f"{host}:{port}",
                test_type='tcp',
                success=success,
                latency_ms=latency_ms if success else None,
                error_message=f"Connection failed: {result}" if not success else None
            )
            
        except Exception as e:
            return ConnectivityTest(
                target=f"{host}:{port}",
                test_type='tcp',
                success=False,
                error_message=str(e)
            )
    
    async def _http_probe(self, url: str) -> ConnectivityTest:
        """Perform HTTP connectivity test"""
        try:
            import aiohttp
            
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    success = response.status < 400
                    
                    return ConnectivityTest(
                        target=url,
                        test_type='http',
                        success=success,
                        latency_ms=latency_ms,
                        error_message=f"HTTP {response.status}" if not success else None
                    )
                    
        except Exception as e:
            return ConnectivityTest(
                target=url,
                test_type='http',
                success=False,
                error_message=str(e)
            )
    
    async def _test_node_connectivity(self, node_id: str):
        """Test connectivity to a specific node"""
        if node_id not in self.node_states:
            return
        
        node_info = self.node_states[node_id]['info']
        
        # Extract connection information
        if 'endpoint' in node_info:
            await self._probe_endpoint(node_info['endpoint'], 'client_node')
        elif 'ip_address' in node_info:
            test_result = await self._ping_probe(node_info['ip_address'])
            
            if node_id not in self.connectivity_history:
                self.connectivity_history[node_id] = []
            
            self.connectivity_history[node_id].append(test_result)
    
    async def _analyze_connectivity_pattern(self, target: str, target_type: str):
        """Analyze connectivity patterns to detect partitions"""
        if target not in self.connectivity_history:
            return
        
        history = self.connectivity_history[target]
        if len(history) < 3:
            return
        
        # Calculate failure rate over recent history
        recent_tests = [test for test in history if 
                      datetime.now() - test.timestamp < timedelta(minutes=10)]
        
        if not recent_tests:
            return
        
        failure_rate = sum(1 for test in recent_tests if not test.success) / len(recent_tests)
        
        # Detect partition based on failure rate
        if failure_rate > 0.7:  # More than 70% failures
            if target_type == 'global_server':
                await self._handle_global_partition()
            elif target_type == 'peer_coordinator':
                await self._handle_peer_partition(target)
            elif target_type == 'client_node':
                await self._handle_node_partition(target, PartitionType.CLIENT_ISOLATION)
    
    async def _consensus_detector(self):
        """Use consensus among nodes to detect partitions"""
        while self.running:
            try:
                # This would implement a consensus algorithm
                # For now, use simple majority voting based on connectivity
                
                await self._perform_consensus_check()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Consensus detector error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_consensus_check(self):
        """Perform consensus-based partition detection"""
        # Count nodes that can reach global server
        global_reachable_count = 0
        total_nodes = 0
        
        for target, history in self.connectivity_history.items():
            if 'global_server' in target or any(server in target for server in self.global_server_endpoints):
                recent_tests = [test for test in history if 
                              datetime.now() - test.timestamp < timedelta(minutes=5)]
                
                if recent_tests:
                    total_nodes += 1
                    success_rate = sum(1 for test in recent_tests if test.success) / len(recent_tests)
                    
                    if success_rate > 0.5:
                        global_reachable_count += 1
        
        # If less than consensus threshold can reach global server, we might be partitioned
        if total_nodes > 0:
            reachability_ratio = global_reachable_count / total_nodes
            
            if reachability_ratio < self.consensus_threshold:
                await self._handle_global_partition()
    
    async def _topology_analyzer(self):
        """Analyze network topology for partition detection"""
        while self.running:
            try:
                await self._analyze_network_topology()
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Topology analyzer error: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_network_topology(self):
        """Analyze network topology changes"""
        # This would implement topology analysis
        # For now, just log the current state
        
        active_nodes = len([node for node, state in self.node_states.items() 
                           if state['status'] == 'active'])
        suspected_nodes = len([node for node, state in self.node_states.items() 
                              if state['status'] == 'suspected'])
        
        self.logger.debug(f"Network topology: {active_nodes} active, {suspected_nodes} suspected nodes")
    
    async def _handle_node_partition(self, node_id: str, partition_type: PartitionType):
        """Handle individual node partition"""
        if node_id in self.node_states:
            self.node_states[node_id]['status'] = 'partitioned'
        
        event_id = f"partition-{node_id}-{int(time.time())}"
        partition_event = PartitionEvent(
            event_id=event_id,
            partition_type=partition_type,
            detected_at=datetime.now(),
            affected_nodes=[node_id],
            severity=0.3,  # Individual node partition is low severity
            recovery_strategy=RecoveryStrategy.WAIT_AND_RETRY
        )
        
        self.active_partitions[event_id] = partition_event
        self.logger.warning(f"Detected node partition: {node_id}")
    
    async def _handle_global_partition(self):
        """Handle partition from global server"""
        event_id = f"global-partition-{int(time.time())}"
        
        # Check if we already have an active global partition
        existing_global = [p for p in self.active_partitions.values() 
                          if p.partition_type == PartitionType.GLOBAL_DISCONNECT]
        
        if existing_global:
            return  # Already handling global partition
        
        partition_event = PartitionEvent(
            event_id=event_id,
            partition_type=PartitionType.GLOBAL_DISCONNECT,
            detected_at=datetime.now(),
            affected_nodes=[self.coordinator_id],
            severity=0.8,  # High severity
            recovery_strategy=RecoveryStrategy.OFFLINE_OPERATION
        )
        
        self.active_partitions[event_id] = partition_event
        self.logger.error("Detected partition from global server - entering offline mode")
    
    async def _handle_peer_partition(self, peer_id: str):
        """Handle partition from peer coordinator"""
        event_id = f"peer-partition-{peer_id}-{int(time.time())}"
        
        partition_event = PartitionEvent(
            event_id=event_id,
            partition_type=PartitionType.REGIONAL_PARTITION,
            detected_at=datetime.now(),
            affected_nodes=[peer_id],
            severity=0.5,  # Medium severity
            recovery_strategy=RecoveryStrategy.AGGRESSIVE_RECONNECT
        )
        
        self.active_partitions[event_id] = partition_event
        self.logger.warning(f"Detected partition from peer coordinator: {peer_id}")
    
    def get_partition_status(self) -> Dict[str, Any]:
        """Get current partition status"""
        return {
            'active_partitions': len(self.active_partitions),
            'partitioned_nodes': [p.affected_nodes for p in self.active_partitions.values()],
            'partition_types': [p.partition_type.value for p in self.active_partitions.values()],
            'node_states': {
                node_id: {
                    'status': state['status'],
                    'last_seen': state['last_seen'].isoformat(),
                    'consecutive_failures': state['consecutive_failures']
                }
                for node_id, state in self.node_states.items()
            },
            'connectivity_summary': {
                target: {
                    'recent_success_rate': self._calculate_recent_success_rate(target),
                    'last_test': history[-1].timestamp.isoformat() if history else None
                }
                for target, history in self.connectivity_history.items()
            }
        }
    
    def _calculate_recent_success_rate(self, target: str) -> float:
        """Calculate recent success rate for a target"""
        if target not in self.connectivity_history:
            return 0.0
        
        recent_tests = [test for test in self.connectivity_history[target] if 
                       datetime.now() - test.timestamp < timedelta(minutes=10)]
        
        if not recent_tests:
            return 0.0
        
        return sum(1 for test in recent_tests if test.success) / len(recent_tests)