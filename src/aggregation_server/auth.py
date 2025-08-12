"""
Authentication and authorization for aggregation server
"""
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..common.config import get_config

# Security scheme
security = HTTPBearer()

# In-memory token storage (use Redis in production)
active_tokens: Dict[str, Dict[str, Any]] = {}
client_credentials: Dict[str, str] = {}


class AuthenticationError(Exception):
    """Authentication error"""
    pass


class AuthorizationError(Exception):
    """Authorization error"""
    pass


def generate_client_token(client_id: str, client_info: Dict[str, Any]) -> str:
    """Generate JWT token for authenticated client"""
    config = get_config()
    
    # Generate secret key if not configured
    secret_key = getattr(config, 'jwt_secret', 'default-secret-key-change-in-production')
    
    payload = {
        'client_id': client_id,
        'client_type': client_info.get('client_type', 'unknown'),
        'issued_at': datetime.utcnow().isoformat(),
        'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat(),
        'capabilities': client_info.get('capabilities', {}),
        'reputation_score': client_info.get('reputation_score', 1.0)
    }
    
    token = jwt.encode(payload, secret_key, algorithm='HS256')
    
    # Store token in memory (use Redis in production)
    active_tokens[token] = {
        'client_id': client_id,
        'issued_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(hours=24),
        'client_info': client_info
    }
    
    return token


def verify_client_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return client information"""
    config = get_config()
    secret_key = getattr(config, 'jwt_secret', 'default-secret-key-change-in-production')
    
    try:
        # Check if token exists in active tokens
        if token not in active_tokens:
            raise AuthenticationError("Token not found or revoked")
        
        token_info = active_tokens[token]
        
        # Check expiration
        if datetime.utcnow() > token_info['expires_at']:
            del active_tokens[token]
            raise AuthenticationError("Token expired")
        
        # Verify JWT
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        
        return {
            'client_id': payload['client_id'],
            'client_type': payload['client_type'],
            'capabilities': payload['capabilities'],
            'reputation_score': payload['reputation_score'],
            'token_info': token_info
        }
    
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")


def revoke_client_token(token: str) -> bool:
    """Revoke client token"""
    if token in active_tokens:
        del active_tokens[token]
        return True
    return False


def generate_api_key(client_id: str) -> str:
    """Generate API key for client"""
    # Create deterministic but secure API key
    key_material = f"{client_id}:{secrets.token_hex(32)}"
    api_key = hashlib.sha256(key_material.encode()).hexdigest()
    
    # Store API key
    client_credentials[client_id] = api_key
    
    return api_key


def verify_api_key(client_id: str, api_key: str) -> bool:
    """Verify API key for client"""
    stored_key = client_credentials.get(client_id)
    return stored_key and secrets.compare_digest(stored_key, api_key)


async def get_authenticated_client(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """FastAPI dependency to get authenticated client ID"""
    try:
        token = credentials.credentials
        client_info = verify_client_token(token)
        return client_info['client_id']
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_client_info(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to get full client information"""
    try:
        token = credentials.credentials
        return verify_client_token(token)
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_client_type(allowed_types: list):
    """Decorator to require specific client types"""
    def decorator(client_info: Dict[str, Any] = Depends(get_client_info)):
        if client_info['client_type'] not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Client type '{client_info['client_type']}' not allowed"
            )
        return client_info
    return decorator


def require_reputation_score(min_score: float):
    """Decorator to require minimum reputation score"""
    def decorator(client_info: Dict[str, Any] = Depends(get_client_info)):
        if client_info['reputation_score'] < min_score:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient reputation score: {client_info['reputation_score']}"
            )
        return client_info
    return decorator


class TokenManager:
    """Token management utilities"""
    
    @staticmethod
    def cleanup_expired_tokens():
        """Remove expired tokens from memory"""
        current_time = datetime.utcnow()
        expired_tokens = [
            token for token, info in active_tokens.items()
            if current_time > info['expires_at']
        ]
        
        for token in expired_tokens:
            del active_tokens[token]
        
        return len(expired_tokens)
    
    @staticmethod
    def get_active_clients() -> list:
        """Get list of currently authenticated clients"""
        current_time = datetime.utcnow()
        return [
            info['client_id'] for info in active_tokens.values()
            if current_time <= info['expires_at']
        ]
    
    @staticmethod
    def get_client_tokens(client_id: str) -> list:
        """Get all active tokens for a client"""
        return [
            token for token, info in active_tokens.items()
            if info['client_id'] == client_id
        ]
    
    @staticmethod
    def revoke_client_tokens(client_id: str) -> int:
        """Revoke all tokens for a client"""
        client_tokens = TokenManager.get_client_tokens(client_id)
        for token in client_tokens:
            del active_tokens[token]
        return len(client_tokens)