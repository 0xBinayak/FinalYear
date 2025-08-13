"""
Real-time visualization of signal characteristics and classification results.
Provides interactive dashboards and plots for demonstration purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import threading
import queue
import time

from src.common.interfaces import SignalSample
from .dataset_integration import LocationProfile, MultiLocationScenario


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    update_interval_ms: int = 100
    max_history_points: int = 1000
    color_scheme: str = "viridis"
    figure_size: Tuple[int, int] = (12, 8)
    enable_animation: bool = True
    save_plots: bool = False
    output_dir: str = "visualization_output"


class SignalVisualization:
    """Real-time visualization of signal processing and federated learning."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data storage for real-time updates
        self.signal_history = queue.Queue(maxsize=self.config.max_history_points)
        self.classification_history = queue.Queue(maxsize=self.config.max_history_points)
        self.location_data = {}
        
        # Animation control
        self.animation_running = False
        self.update_thread = None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.config.color_scheme)
    
    def create_signal_constellation_plot(self, samples: List[SignalSample], 
                                       title: str = "Signal Constellation Diagrams") -> go.Figure:
        """Create interactive constellation diagrams for different modulation types."""
        
        # Group samples by modulation type
        modulation_groups = {}
        for sample in samples[:100]:  # Limit for performance
            mod_type = sample.modulation_type
            if mod_type not in modulation_groups:
                modulation_groups[mod_type] = []
            modulation_groups[mod_type].append(sample)
        
        # Create subplots
        num_mods = len(modulation_groups)
        cols = min(3, num_mods)
        rows = (num_mods + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(modulation_groups.keys()),
            specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
        )
        
        colors = px.colors.qualitative.Set1
        
        for idx, (mod_type, mod_samples) in enumerate(modulation_groups.items()):
            row = idx // cols + 1
            col = idx % cols + 1
            
            # Extract I/Q data
            i_data = []
            q_data = []
            snr_values = []
            
            for sample in mod_samples[:50]:  # Limit points per modulation
                iq = sample.iq_data
                i_data.extend(iq.real)
                q_data.extend(iq.imag)
                snr_values.extend([sample.snr] * len(iq))
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=i_data[::10],  # Subsample for performance
                    y=q_data[::10],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=snr_values[::10],
                        colorscale='Viridis',
                        showscale=True if idx == 0 else False,
                        colorbar=dict(title="SNR (dB)") if idx == 0 else None
                    ),
                    name=mod_type,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            height=300 * rows,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(title_text="In-phase", row=i, col=j)
                fig.update_yaxes(title_text="Quadrature", row=i, col=j)
        
        return fig
    
    def create_signal_spectrogram(self, sample: SignalSample, 
                                title: str = "Signal Spectrogram") -> go.Figure:
        """Create spectrogram visualization of signal sample."""
        
        # Compute spectrogram
        from scipy import signal as scipy_signal
        
        iq_data = sample.iq_data
        fs = sample.sample_rate
        
        f, t, Sxx = scipy_signal.spectrogram(
            iq_data, fs=fs, nperseg=256, noverlap=128
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=Sxx_db,
            x=t,
            y=f / 1e6,  # Convert to MHz
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)")
        ))
        
        fig.update_layout(
            title=f"{title} - {sample.modulation_type} (SNR: {sample.snr:.1f} dB)",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (MHz)",
            height=400
        )
        
        return fig
    
    def create_location_map(self, scenario: MultiLocationScenario, 
                          client_data: Dict[str, Any] = None) -> go.Figure:
        """Create interactive map showing client locations and signal quality."""
        
        # Prepare location data
        locations = []
        for location in scenario.locations:
            num_clients = scenario.client_distribution.get(location.name, 0)
            
            # Get average signal quality if available
            avg_snr = 0
            if client_data and location.name in client_data:
                snr_values = [sample.snr for sample in client_data[location.name]]
                avg_snr = np.mean(snr_values) if snr_values else 0
            
            locations.append({
                'name': location.name,
                'lat': location.latitude,
                'lon': location.longitude,
                'environment': location.environment_type,
                'clients': num_clients,
                'avg_snr': avg_snr,
                'noise_floor': location.noise_floor_db,
                'interference_sources': len(location.interference_sources)
            })
        
        df = pd.DataFrame(locations)
        
        # Create map
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            size="clients",
            color="avg_snr",
            hover_name="name",
            hover_data={
                "environment": True,
                "clients": True,
                "avg_snr": ":.1f",
                "noise_floor": True,
                "interference_sources": True
            },
            color_continuous_scale="RdYlGn",
            size_max=30,
            zoom=3,
            title="Federated Learning Client Locations"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            coloraxis_colorbar=dict(title="Average SNR (dB)")
        )
        
        return fig
    
    def create_classification_accuracy_plot(self, accuracy_history: List[Dict[str, Any]]) -> go.Figure:
        """Create real-time classification accuracy plot."""
        
        df = pd.DataFrame(accuracy_history)
        
        fig = go.Figure()
        
        # Add traces for different modulation types
        if 'modulation_accuracies' in df.columns:
            modulation_types = set()
            for accuracies in df['modulation_accuracies']:
                if isinstance(accuracies, dict):
                    modulation_types.update(accuracies.keys())
            
            for mod_type in modulation_types:
                accuracies = []
                timestamps = []
                
                for _, row in df.iterrows():
                    if isinstance(row['modulation_accuracies'], dict):
                        acc = row['modulation_accuracies'].get(mod_type, 0)
                        accuracies.append(acc)
                        timestamps.append(row.get('timestamp', len(accuracies)))
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=accuracies,
                    mode='lines+markers',
                    name=mod_type,
                    line=dict(width=2)
                ))
        
        # Add overall accuracy if available
        if 'overall_accuracy' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['overall_accuracy'],
                mode='lines+markers',
                name='Overall',
                line=dict(width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Real-time Classification Accuracy",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_federated_learning_progress(self, training_history: List[Dict[str, Any]]) -> go.Figure:
        """Create federated learning training progress visualization."""
        
        df = pd.DataFrame(training_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Model Accuracy", "Training Loss", 
                "Client Participation", "Communication Rounds"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if not df.empty:
            # Model accuracy
            if 'accuracy' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['accuracy'], name="Accuracy"),
                    row=1, col=1
                )
            
            # Training loss
            if 'loss' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['loss'], name="Loss"),
                    row=1, col=2
                )
            
            # Client participation
            if 'active_clients' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['active_clients'], name="Active Clients"),
                    row=2, col=1
                )
            
            # Communication rounds
            if 'round_number' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['round_number'], name="Round"),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Federated Learning Training Progress",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_signal_quality_heatmap(self, location_data: Dict[str, List[SignalSample]]) -> go.Figure:
        """Create heatmap showing signal quality across locations and time."""
        
        # Prepare data for heatmap
        locations = list(location_data.keys())
        time_slots = []
        quality_matrix = []
        
        # Create time slots (hourly for 24 hours)
        for hour in range(24):
            time_slots.append(f"{hour:02d}:00")
        
        for location in locations:
            samples = location_data[location]
            hourly_quality = []
            
            for hour in range(24):
                # Filter samples for this hour
                hour_samples = [
                    s for s in samples 
                    if s.timestamp and s.timestamp.hour == hour
                ]
                
                if hour_samples:
                    avg_snr = np.mean([s.snr for s in hour_samples])
                    hourly_quality.append(avg_snr)
                else:
                    hourly_quality.append(0)
            
            quality_matrix.append(hourly_quality)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=quality_matrix,
            x=time_slots,
            y=locations,
            colorscale='RdYlGn',
            colorbar=dict(title="Average SNR (dB)")
        ))
        
        fig.update_layout(
            title="Signal Quality Heatmap (24-hour period)",
            xaxis_title="Time of Day",
            yaxis_title="Location",
            height=400
        )
        
        return fig
    
    def create_concept_drift_visualization(self, drift_data: List[Dict[str, Any]]) -> go.Figure:
        """Visualize concept drift detection and adaptation."""
        
        df = pd.DataFrame(drift_data)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Drift Detection Score", "Model Performance"],
            shared_xaxes=True
        )
        
        if not df.empty:
            # Drift detection score
            if 'drift_score' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['drift_score'], 
                        name="Drift Score",
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                # Add threshold line
                if 'drift_threshold' in df.columns:
                    threshold = df['drift_threshold'].iloc[0] if len(df) > 0 else 0.5
                    fig.add_hline(
                        y=threshold, 
                        line_dash="dash", 
                        line_color="orange",
                        annotation_text="Drift Threshold",
                        row=1, col=1
                    )
            
            # Model performance
            if 'accuracy_before_adaptation' in df.columns and 'accuracy_after_adaptation' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['accuracy_before_adaptation'], 
                        name="Before Adaptation",
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['accuracy_after_adaptation'], 
                        name="After Adaptation",
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Concept Drift Detection and Adaptation",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Drift Score", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        
        return fig
    
    def create_comparison_dashboard(self, federated_results: Dict[str, Any], 
                                  centralized_results: Dict[str, Any]) -> go.Figure:
        """Create comparison dashboard between federated and centralized learning."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Accuracy Comparison", "Training Time", 
                "Communication Cost", "Privacy Preservation"
            ]
        )
        
        # Accuracy comparison
        methods = ['Federated', 'Centralized']
        accuracies = [
            federated_results.get('final_accuracy', 0),
            centralized_results.get('final_accuracy', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=methods, y=accuracies, name="Accuracy"),
            row=1, col=1
        )
        
        # Training time comparison
        training_times = [
            federated_results.get('training_time_hours', 0),
            centralized_results.get('training_time_hours', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=methods, y=training_times, name="Training Time"),
            row=1, col=2
        )
        
        # Communication cost
        comm_costs = [
            federated_results.get('communication_mb', 0),
            centralized_results.get('communication_mb', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=methods, y=comm_costs, name="Communication"),
            row=2, col=1
        )
        
        # Privacy preservation (qualitative)
        privacy_scores = [
            federated_results.get('privacy_score', 0.9),  # High for federated
            centralized_results.get('privacy_score', 0.1)  # Low for centralized
        ]
        
        fig.add_trace(
            go.Bar(x=methods, y=privacy_scores, name="Privacy Score"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Federated vs Centralized Learning Comparison",
            height=600,
            showlegend=False
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Hours", row=1, col=2)
        fig.update_yaxes(title_text="MB", row=2, col=1)
        fig.update_yaxes(title_text="Score (0-1)", row=2, col=2)
        
        return fig
    
    def start_real_time_monitoring(self, update_callback=None):
        """Start real-time monitoring with live updates."""
        
        self.animation_running = True
        
        def update_loop():
            while self.animation_running:
                try:
                    if update_callback:
                        update_callback()
                    time.sleep(self.config.update_interval_ms / 1000.0)
                except Exception as e:
                    self.logger.error(f"Error in real-time update: {e}")
        
        self.update_thread = threading.Thread(target=update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Started real-time monitoring")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        
        self.animation_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        self.logger.info("Stopped real-time monitoring")
    
    def save_visualization(self, fig: go.Figure, filename: str):
        """Save visualization to file."""
        
        if self.config.save_plots:
            import os
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            filepath = os.path.join(self.config.output_dir, filename)
            
            # Save as HTML for interactive plots
            fig.write_html(f"{filepath}.html")
            
            # Save as PNG for static plots
            try:
                fig.write_image(f"{filepath}.png")
            except Exception as e:
                self.logger.warning(f"Could not save PNG: {e}")
            
            self.logger.info(f"Saved visualization: {filepath}")
    
    def create_comprehensive_dashboard(self, demo_data: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create comprehensive dashboard with all visualizations."""
        
        dashboard = {}
        
        # Signal constellation plots
        if 'signal_samples' in demo_data:
            dashboard['constellation'] = self.create_signal_constellation_plot(
                demo_data['signal_samples']
            )
        
        # Location map
        if 'scenario' in demo_data:
            dashboard['location_map'] = self.create_location_map(
                demo_data['scenario'],
                demo_data.get('location_data')
            )
        
        # Classification accuracy
        if 'accuracy_history' in demo_data:
            dashboard['accuracy'] = self.create_classification_accuracy_plot(
                demo_data['accuracy_history']
            )
        
        # Training progress
        if 'training_history' in demo_data:
            dashboard['training_progress'] = self.create_federated_learning_progress(
                demo_data['training_history']
            )
        
        # Signal quality heatmap
        if 'location_data' in demo_data:
            dashboard['quality_heatmap'] = self.create_signal_quality_heatmap(
                demo_data['location_data']
            )
        
        # Concept drift
        if 'drift_data' in demo_data:
            dashboard['concept_drift'] = self.create_concept_drift_visualization(
                demo_data['drift_data']
            )
        
        # Comparison
        if 'federated_results' in demo_data and 'centralized_results' in demo_data:
            dashboard['comparison'] = self.create_comparison_dashboard(
                demo_data['federated_results'],
                demo_data['centralized_results']
            )
        
        return dashboard