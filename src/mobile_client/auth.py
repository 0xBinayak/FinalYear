"""
Authentication and registration for mobile clients
"""
import hashlib
import json
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid

from ..common.interfaces import ClientInfo


@dataclass
class MobileAuthConfig:
    """Mobile authentication configuration"""
    server_url: str
    client_id: str
    device_fingerprint: str
    auto_refresh_token: bool = True
    token_refresh_threshold_minutes: int = 60
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MobileAuthenticator:
    """Mobile client authentication manager"""
    
    def __init__(self, config: MobileAuthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Authentication state
        self.auth_token = None
        self.token_expires_at = None
        self.refresh_token = None
        self.is_authenticated = False
        
        # Device information
        self.device_info = self._generate_device_info()
        
        # Session management
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'FederatedMobileClient/1.0 ({self.device_info["platform"]})',
            'Content-Type': 'application/json'
        })
    
    def _generate_device_info(self) -> Dict[str, Any]:
        """Generate device information for authentication"""
        import platform
        import psutil
        
        try:
            # Get system information
            system_info = {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'hostname': platform.node(),
                'python_version': platform.python_version()
            }
            
            # Get hardware information
            try:
                memory = psutil.virtual_memory()
                cpu_count = psutil.cpu_count()
                
                hardware_info = {
                    'cpu_count': cpu_count,
                    'memory_total_gb': round(memory.total / (1024**3), 2),
                    'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
                }
            except:
                hardware_info = {'cpu_count': 1, 'memory_total_gb': 1.0}
            
            return {**system_info, **hardware_info}
            
        except Exception as e:
            self.logger.warning(f"Could not gather device info: {e}")
            return {
                'platform': 'unknown',
                'platform_version': 'unknown',
                'architecture': 'unknown'
            }
    
    def _generate_device_fingerprint(self) -> str:
        """Generate unique device fingerprint"""
        # Create fingerprint from device characteristics
        fingerprint_data = {
            'client_id': self.config.client_id,
            'platform': self.device_info.get('platform', 'unknown'),
            'architecture': self.device_info.get('architecture', 'unknown'),
            'cpu_count': self.device_info.get('cpu_count', 1),
            'memory_total_gb': self.device_info.get('memory_total_gb', 1.0)
        }
        
        # Create hash
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]
    
    def register(self, client_info: ClientInfo) -> Tuple[bool, str]:
        """Register mobile client with server"""
        try:
            self.logger.info(f"Registering mobile client {self.config.client_id}")
            
            # Prepare registration data
            registration_data = {
                'client_info': asdict(client_info),
                'device_info': self.device_info,
                'device_fingerprint': self.config.device_fingerprint,
                'registration_timestamp': datetime.now().isoformat(),
                'client_version': '1.0',
                'capabilities': client_info.capabilities
            }
            
            # Make registration request
            response = self._make_request(
                'POST',
                f"{self.config.server_url}/api/v1/clients/register",
                json=registration_data
            )
            
            if response and response.status_code == 200:
                result = response.json()
                
                # Store authentication tokens
                self.auth_token = result.get('access_token')
                self.refresh_token = result.get('refresh_token')
                
                # Parse token expiration
                expires_in = result.get('expires_in', 3600)  # Default 1 hour
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                self.is_authenticated = True
                
                self.logger.info("Mobile client registered successfully")
                return True, "Registration successful"
            
            elif response:
                error_msg = f"Registration failed: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('detail', response.text)}"
                    except:
                        error_msg += f" - {response.text}"
                
                self.logger.error(error_msg)
                return False, error_msg
            
            else:
                return False, "Network error during registration"
                
        except Exception as e:
            error_msg = f"Registration error: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def authenticate(self, username: Optional[str] = None, 
                    password: Optional[str] = None) -> Tuple[bool, str]:
        """Authenticate with server using credentials or device fingerprint"""
        try:
            self.logger.info("Authenticating mobile client")
            
            # Prepare authentication data
            auth_data = {
                'client_id': self.config.client_id,
                'device_fingerprint': self.config.device_fingerprint,
                'device_info': self.device_info,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add credentials if provided
            if username and password:
                auth_data['username'] = username
                auth_data['password'] = hashlib.sha256(password.encode()).hexdigest()
            
            # Make authentication request
            response = self._make_request(
                'POST',
                f"{self.config.server_url}/api/v1/auth/token",
                json=auth_data
            )
            
            if response and response.status_code == 200:
                result = response.json()
                
                # Store tokens
                self.auth_token = result.get('access_token')
                self.refresh_token = result.get('refresh_token')
                
                # Parse expiration
                expires_in = result.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                self.is_authenticated = True
                
                self.logger.info("Authentication successful")
                return True, "Authentication successful"
            
            elif response:
                error_msg = f"Authentication failed: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('detail', response.text)}"
                    except:
                        error_msg += f" - {response.text}"
                
                self.logger.error(error_msg)
                return False, error_msg
            
            else:
                return False, "Network error during authentication"
                
        except Exception as e:
            error_msg = f"Authentication error: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def refresh_access_token(self) -> Tuple[bool, str]:
        """Refresh access token using refresh token"""
        if not self.refresh_token:
            return False, "No refresh token available"
        
        try:
            self.logger.info("Refreshing access token")
            
            refresh_data = {
                'refresh_token': self.refresh_token,
                'client_id': self.config.client_id,
                'device_fingerprint': self.config.device_fingerprint
            }
            
            response = self._make_request(
                'POST',
                f"{self.config.server_url}/api/v1/auth/refresh",
                json=refresh_data
            )
            
            if response and response.status_code == 200:
                result = response.json()
                
                # Update tokens
                self.auth_token = result.get('access_token')
                if 'refresh_token' in result:
                    self.refresh_token = result['refresh_token']
                
                # Update expiration
                expires_in = result.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                self.logger.info("Token refreshed successfully")
                return True, "Token refreshed"
            
            else:
                self.is_authenticated = False
                error_msg = f"Token refresh failed: {response.status_code if response else 'Network error'}"
                self.logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            self.is_authenticated = False
            error_msg = f"Token refresh error: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def is_token_valid(self) -> bool:
        """Check if current token is valid"""
        if not self.auth_token or not self.token_expires_at:
            return False
        
        # Check if token expires soon (within threshold)
        threshold = timedelta(minutes=self.config.token_refresh_threshold_minutes)
        return datetime.now() + threshold < self.token_expires_at
    
    def ensure_authenticated(self) -> bool:
        """Ensure client is authenticated, refresh token if needed"""
        if not self.is_authenticated:
            return False
        
        if self.is_token_valid():
            return True
        
        # Try to refresh token
        if self.config.auto_refresh_token and self.refresh_token:
            success, _ = self.refresh_access_token()
            return success
        
        return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        if not self.ensure_authenticated():
            return {}
        
        return {
            'Authorization': f'Bearer {self.auth_token}',
            'X-Client-ID': self.config.client_id,
            'X-Device-Fingerprint': self.config.device_fingerprint
        }
    
    def logout(self) -> bool:
        """Logout and invalidate tokens"""
        try:
            if self.auth_token:
                # Notify server of logout
                headers = self.get_auth_headers()
                self._make_request(
                    'POST',
                    f"{self.config.server_url}/api/v1/auth/logout",
                    headers=headers
                )
            
            # Clear local state
            self.auth_token = None
            self.refresh_token = None
            self.token_expires_at = None
            self.is_authenticated = False
            
            self.logger.info("Logged out successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Logout error: {e}")
            return False
    
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.config.max_retry_attempts):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    self.logger.error(f"All {self.config.max_retry_attempts} request attempts failed")
        
        return None
    
    def get_authentication_status(self) -> Dict[str, Any]:
        """Get current authentication status"""
        return {
            'is_authenticated': self.is_authenticated,
            'has_auth_token': bool(self.auth_token),
            'has_refresh_token': bool(self.refresh_token),
            'token_expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None,
            'token_valid': self.is_token_valid(),
            'client_id': self.config.client_id,
            'device_fingerprint': self.config.device_fingerprint,
            'server_url': self.config.server_url
        }
    
    def cleanup(self):
        """Cleanup authentication resources"""
        self.logout()
        self.session.close()
        self.logger.info("Authentication manager cleaned up")


def generate_client_id() -> str:
    """Generate unique client ID for mobile device"""
    # Use UUID4 for unique client ID
    return f"mobile_{uuid.uuid4().hex[:16]}"


def create_mobile_auth_config(server_url: str, client_id: Optional[str] = None) -> MobileAuthConfig:
    """Create mobile authentication configuration"""
    if client_id is None:
        client_id = generate_client_id()
    
    # Generate device fingerprint
    temp_config = MobileAuthConfig(
        server_url=server_url,
        client_id=client_id,
        device_fingerprint=""
    )
    
    # Create authenticator to generate fingerprint
    temp_auth = MobileAuthenticator(temp_config)
    device_fingerprint = temp_auth._generate_device_fingerprint()
    
    return MobileAuthConfig(
        server_url=server_url,
        client_id=client_id,
        device_fingerprint=device_fingerprint
    )