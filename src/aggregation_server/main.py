"""
FastAPI-based aggregation server for federated learning
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .server import AggregationServer
from .models import (
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    ModelUpdateRequest,
    ModelUpdateResponse,
    GlobalModelResponse,
    HealthResponse
)
from .auth import get_authenticated_client
from ..common.config import get_config
from ..common.interfaces import ClientInfo, ModelUpdate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global server instance
aggregation_server: Optional[AggregationServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global aggregation_server
    
    # Startup
    config = get_config()
    aggregation_server = AggregationServer(config)
    await aggregation_server.initialize()
    logger.info("Aggregation server initialized")
    
    yield
    
    # Shutdown
    if aggregation_server:
        await aggregation_server.shutdown()
    logger.info("Aggregation server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Federated Learning Aggregation Server",
    description="Central aggregation service for federated learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    health_status = await aggregation_server.get_health_status()
    return HealthResponse(**health_status)


@app.post("/api/v1/clients/register", response_model=ClientRegistrationResponse)
async def register_client(request: ClientRegistrationRequest):
    """Register a new client"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        client_info = ClientInfo(
            client_id=request.client_id,
            client_type=request.client_type,
            capabilities=request.capabilities,
            location=request.location,
            network_info=request.network_info,
            hardware_specs=request.hardware_specs,
            reputation_score=1.0
        )
        
        token = await aggregation_server.register_client(client_info)
        return ClientRegistrationResponse(
            client_id=request.client_id,
            token=token,
            status="registered"
        )
    
    except Exception as e:
        logger.error(f"Client registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )


@app.post("/api/v1/models/update", response_model=ModelUpdateResponse)
async def submit_model_update(
    request: ModelUpdateRequest,
    client_id: str = Depends(get_authenticated_client)
):
    """Submit model update from client"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        model_update = ModelUpdate(
            client_id=client_id,
            model_weights=request.model_weights,
            training_metrics=request.training_metrics,
            data_statistics=request.data_statistics,
            computation_time=request.computation_time,
            network_conditions=request.network_conditions,
            privacy_budget_used=request.privacy_budget_used
        )
        
        success = await aggregation_server.receive_model_update(client_id, model_update)
        
        if success:
            return ModelUpdateResponse(
                client_id=client_id,
                status="accepted",
                round_number=await aggregation_server.get_current_round()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model update rejected"
            )
    
    except Exception as e:
        logger.error(f"Model update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Update failed: {str(e)}"
        )


@app.get("/api/v1/models/global", response_model=GlobalModelResponse)
async def get_global_model(client_id: str = Depends(get_authenticated_client)):
    """Get the latest global model"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        global_model = await aggregation_server.get_global_model(client_id)
        
        if global_model:
            return GlobalModelResponse(
                model_weights=global_model["weights"],
                version=global_model["version"],
                round_number=global_model["round"],
                training_config=global_model["config"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No global model available"
            )
    
    except Exception as e:
        logger.error(f"Global model retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retrieval failed: {str(e)}"
        )


@app.get("/api/v1/training/config")
async def get_training_config(client_id: str = Depends(get_authenticated_client)):
    """Get training configuration for client"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        config = await aggregation_server.get_training_configuration(client_id)
        return config
    
    except Exception as e:
        logger.error(f"Training config retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config retrieval failed: {str(e)}"
        )


@app.post("/api/v1/clients/{client_id}/metrics")
async def report_client_metrics(
    client_id: str,
    metrics: Dict,
    authenticated_client_id: str = Depends(get_authenticated_client)
):
    """Report client metrics"""
    if client_id != authenticated_client_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot report metrics for other clients"
        )
    
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        await aggregation_server.report_client_metrics(client_id, metrics)
        return {"status": "metrics_recorded"}
    
    except Exception as e:
        logger.error(f"Metrics reporting failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Metrics reporting failed: {str(e)}"
        )


@app.get("/api/v1/status")
async def get_server_status():
    """Get server status and statistics"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        status_info = await aggregation_server.get_server_status()
        return status_info
    
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status retrieval failed: {str(e)}"
        )


@app.get("/api/v1/aggregation/strategies")
async def get_available_strategies():
    """Get available aggregation strategies"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        strategies = await aggregation_server.get_available_strategies()
        return {"strategies": strategies}
    
    except Exception as e:
        logger.error(f"Strategy retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy retrieval failed: {str(e)}"
        )


@app.get("/api/v1/convergence/history")
async def get_convergence_history():
    """Get convergence history"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        history = await aggregation_server.get_convergence_history()
        return {"convergence_history": history}
    
    except Exception as e:
        logger.error(f"Convergence history retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Convergence history retrieval failed: {str(e)}"
        )


@app.post("/api/v1/aggregation/strategy")
async def update_aggregation_strategy(
    strategy: str,
    client_id: str = Depends(get_authenticated_client)
):
    """Update aggregation strategy (admin only)"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        # Check if client has admin privileges (simplified check)
        client_info = aggregation_server.registered_clients.get(client_id)
        if not client_info or client_info.reputation_score < 0.9:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges"
            )
        
        success = await aggregation_server.update_aggregation_strategy(strategy)
        
        if success:
            return {"status": "strategy_updated", "new_strategy": strategy}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid strategy or update failed"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy update failed: {str(e)}"
        )


@app.get("/api/v1/security/status")
async def get_security_status(client_id: str = Depends(get_authenticated_client)):
    """Get security and privacy status"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        # Check if client has sufficient privileges
        client_info = aggregation_server.registered_clients.get(client_id)
        if not client_info or client_info.reputation_score < 0.8:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges for security status"
            )
        
        security_status = await aggregation_server.get_security_status()
        return security_status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security status retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security status retrieval failed: {str(e)}"
        )


@app.get("/api/v1/privacy/budget")
async def get_privacy_budget(client_id: str = Depends(get_authenticated_client)):
    """Get privacy budget status for client"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        budget_status = await aggregation_server.get_client_privacy_budget(client_id)
        return budget_status
    
    except Exception as e:
        logger.error(f"Privacy budget retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Privacy budget retrieval failed: {str(e)}"
        )


@app.get("/api/v1/compliance/report")
async def get_compliance_report(
    days: int = 30,
    client_id: str = Depends(get_authenticated_client)
):
    """Get compliance report (admin only)"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        # Check admin privileges
        client_info = aggregation_server.registered_clients.get(client_id)
        if not client_info or client_info.reputation_score < 0.95:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        report = await aggregation_server.generate_compliance_report(days)
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compliance report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance report generation failed: {str(e)}"
        )


@app.post("/api/v1/security/block-client")
async def block_client(
    target_client_id: str,
    reason: str,
    client_id: str = Depends(get_authenticated_client)
):
    """Block a client (admin only)"""
    if not aggregation_server:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    try:
        # Check admin privileges
        client_info = aggregation_server.registered_clients.get(client_id)
        if not client_info or client_info.reputation_score < 0.95:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        success = await aggregation_server.block_client(target_client_id, reason)
        
        if success:
            return {"status": "client_blocked", "client_id": target_client_id, "reason": reason}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to block client"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Client blocking failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Client blocking failed: {str(e)}"
        )


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "main:app",
        host=config.network.host,
        port=config.network.port,
        reload=config.debug,
        log_level=config.monitoring.log_level.lower()
    )