"""
Vehicular Simulation Module

Contains:
- Base Station class
- Kalman Filter for trajectory prediction
- Vehicle types and profiles
- VehicularClient class with physics-based RSSI and contact time
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from .config import ExperimentConfig


# =============================================================================
# Base Station
# =============================================================================

@dataclass
class BaseStation:
    """RSU/Base Station"""
    id: int
    position: np.ndarray
    coverage_radius: float
    tx_power: float
    
    def distance_to(self, pos: np.ndarray) -> float:
        return np.linalg.norm(self.position - pos)
    
    def is_in_coverage(self, pos: np.ndarray) -> bool:
        return self.distance_to(pos) <= self.coverage_radius


def create_base_stations(config: ExperimentConfig) -> List[BaseStation]:
    """Create base stations in a grid pattern"""
    return [
        BaseStation(0, np.array([config.grid_width*0.25, config.grid_height*0.25]), 
                   config.bs_coverage_radius, config.bs_tx_power),
        BaseStation(1, np.array([config.grid_width*0.75, config.grid_height*0.25]), 
                   config.bs_coverage_radius, config.bs_tx_power),
        BaseStation(2, np.array([config.grid_width*0.25, config.grid_height*0.75]), 
                   config.bs_coverage_radius, config.bs_tx_power),
        BaseStation(3, np.array([config.grid_width*0.75, config.grid_height*0.75]), 
                   config.bs_coverage_radius, config.bs_tx_power),
    ]


# =============================================================================
# Kalman Filter (FLIPS Section IV.G)
# =============================================================================

class KalmanFilter2D:
    """Kalman Filter for position/velocity estimation and prediction"""
    
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        
        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        q = 0.5
        self.Q = np.diag([q, q, q*0.1, q*0.1])
        
        # Measurement noise (GPS accuracy)
        r = 5.0
        self.R = np.diag([r, r])
        
        # State and covariance
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100
        
    def initialize(self, position: np.ndarray, velocity: np.ndarray):
        self.x = np.array([position[0], position[1], velocity[0], velocity[1]])
        self.P = np.eye(4) * 10
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()
    
    def update(self, measurement: np.ndarray):
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()
    
    def get_position(self) -> np.ndarray:
        return self.x[:2]
    
    def get_velocity(self) -> np.ndarray:
        return self.x[2:]
    
    def get_speed(self) -> float:
        return np.linalg.norm(self.get_velocity())


# =============================================================================
# Vehicle Types
# =============================================================================

class VehicleType(Enum):
    TAXI = "taxi"
    BUS = "bus"
    DELIVERY = "delivery"
    COMMUTER = "commuter"


VEHICLE_PROFILES = {
    VehicleType.TAXI: {
        'speed_range': (5.0, 15.0),
        'direction_change_prob': 0.15,
        'stop_prob': 0.10,
        'entropy_range': (0.7, 0.95),
        'resource_factor': 0.85,
    },
    VehicleType.BUS: {
        'speed_range': (8.0, 12.0),
        'direction_change_prob': 0.02,
        'stop_prob': 0.15,
        'entropy_range': (0.6, 0.85),
        'resource_factor': 0.95,
    },
    VehicleType.DELIVERY: {
        'speed_range': (6.0, 14.0),
        'direction_change_prob': 0.08,
        'stop_prob': 0.20,
        'entropy_range': (0.4, 0.7),
        'resource_factor': 0.80,
    },
    VehicleType.COMMUTER: {
        'speed_range': (8.0, 20.0),
        'direction_change_prob': 0.05,
        'stop_prob': 0.03,
        'entropy_range': (0.3, 0.6),
        'resource_factor': 0.90,
    },
}


# =============================================================================
# Client State (for Markov model)
# =============================================================================

class ClientState(Enum):
    """Possible outcomes of a training round"""
    SUCCESS = "S"
    CONNECTION_FAILURE = "C"
    RESOURCE_FAILURE = "R"
    ABORT = "A"


# =============================================================================
# Vehicular Client
# =============================================================================

class VehicularClient:
    """
    FL Client with realistic vehicular simulation.
    
    Contains:
    - Physical position/velocity
    - Path loss-based RSSI
    - Kalman filter for prediction
    - Local dataset indices
    """
    
    def __init__(self, client_id: int, config: ExperimentConfig,
                 base_stations: List[BaseStation],
                 data_indices: List[int],
                 vehicle_type: VehicleType = None):
        self.client_id = client_id
        self.config = config
        self.base_stations = base_stations
        self.data_indices = data_indices
        
        # Vehicle type (random if not specified)
        if vehicle_type is None:
            type_probs = [0.15, 0.10, 0.20, 0.55]  # taxi, bus, delivery, commuter
            vehicle_type = np.random.choice(list(VehicleType), p=type_probs)
        self.vehicle_type = vehicle_type
        self.profile = VEHICLE_PROFILES[vehicle_type]
        
        # Initialize position randomly
        self.position = np.array([
            random.uniform(0, config.grid_width),
            random.uniform(0, config.grid_height)
        ])
        
        # Initialize velocity
        speed = random.uniform(*self.profile['speed_range'])
        direction = random.uniform(0, 2 * np.pi)
        self.velocity = np.array([
            speed * np.cos(direction),
            speed * np.sin(direction)
        ])
        
        # Kalman filter
        self.kalman = KalmanFilter2D(dt=config.position_update_interval)
        self.kalman.initialize(self.position, self.velocity)
        
        # Connection state
        self.connected_bs: Optional[BaseStation] = None
        self.time_in_current_bs = 0.0
        self._update_connection()
        
        # Resource level (battery/compute)
        self.resource_level = random.uniform(0.7, 1.0) * self.profile['resource_factor']
        
        # Data entropy (based on vehicle type)
        self.base_entropy = random.uniform(*self.profile['entropy_range'])
        
        # State tracking
        self.is_stopped = False
        self.stop_timer = 0.0
        
        # History tracking (for Markov and metrics)
        self.outcome_history: List[ClientState] = []
        self.accuracy_history: List[float] = []
        self.total_dropouts = 0
        self.consecutive_failures = 0
        
    def _update_connection(self):
        """Find best base station connection"""
        best_bs = None
        best_rssi = -np.inf
        
        for bs in self.base_stations:
            rssi = self._calculate_rssi(bs)
            if rssi > best_rssi:
                best_rssi = rssi
                best_bs = bs
        
        # Check for handover
        if self.connected_bs != best_bs:
            self.time_in_current_bs = 0.0
        
        self.connected_bs = best_bs if best_rssi > self.config.rssi_min_connection else None
        
    def _calculate_rssi(self, bs: BaseStation) -> float:
        """Calculate RSSI using path loss model"""
        distance = bs.distance_to(self.position)
        if distance < 1.0:
            distance = 1.0
            
        # Log-distance path loss
        path_loss = (self.config.reference_path_loss + 
                    10 * self.config.path_loss_exponent * np.log10(distance))
        
        # Shadow fading
        fading = np.random.normal(0, self.config.shadow_fading_std)
        
        rssi = bs.tx_power - path_loss + fading
        return rssi
    
    def get_rssi(self) -> float:
        """Get current RSSI to connected BS"""
        if self.connected_bs is None:
            return -120.0
        return self._calculate_rssi(self.connected_bs)
    
    def get_normalized_rssi(self) -> float:
        """Normalize RSSI to [0, 1]"""
        rssi = self.get_rssi()
        rssi_min = self.config.rssi_min_connection
        rssi_max = self.config.rssi_good_connection
        normalized = (rssi - rssi_min) / (rssi_max - rssi_min)
        return np.clip(normalized, 0.0, 1.0)
    
    def get_contact_time(self) -> float:
        """
        Calculate predicted contact time with current BS.
        Uses geometric intersection of trajectory with coverage circle.
        """
        if self.connected_bs is None:
            return 0.0
            
        pos = self.kalman.get_position()
        vel = self.kalman.get_velocity()
        speed = np.linalg.norm(vel)
        
        if speed < 0.1:  # Stopped
            return float('inf') if self.connected_bs.is_in_coverage(pos) else 0.0
        
        bs = self.connected_bs
        d = pos - bs.position  # Vector from BS to vehicle
        r = bs.coverage_radius
        
        # Solve quadratic: ||d + v*t|| = r
        a = np.dot(vel, vel)
        b = 2 * np.dot(d, vel)
        c = np.dot(d, d) - r**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return 0.0
            
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Take positive solution
        times = [t for t in [t1, t2] if t > 0]
        if not times:
            return 0.0
            
        return min(times)
    
    def get_normalized_contact_time(self) -> float:
        """Normalize contact time to [0, 1]"""
        ct = self.get_contact_time()
        if ct == float('inf'):
            return 1.0
        return min(1.0, ct / (self.config.round_duration * 3))
    
    def update_position(self, dt: float):
        """Update vehicle position for one time step"""
        if self.is_stopped:
            self.stop_timer -= dt
            if self.stop_timer <= 0:
                self.is_stopped = False
            else:
                # Recharge while stopped
                self.resource_level = min(1.0, self.resource_level + 0.005 * dt)
                return
        
        # Check for stop
        if random.random() < self.profile['stop_prob'] * dt:
            self.is_stopped = True
            self.stop_timer = random.expovariate(1/30.0)
            return
        
        # Check for direction change
        if random.random() < self.profile['direction_change_prob']:
            turn = np.random.normal(0, np.pi/4)
            speed = np.linalg.norm(self.velocity)
            current_dir = np.arctan2(self.velocity[1], self.velocity[0])
            new_dir = current_dir + turn
            self.velocity = np.array([
                speed * np.cos(new_dir),
                speed * np.sin(new_dir)
            ])
        
        # Update position
        new_pos = self.position + self.velocity * dt
        
        # Wrap around boundaries
        new_pos[0] = new_pos[0] % self.config.grid_width
        new_pos[1] = new_pos[1] % self.config.grid_height
        
        self.position = new_pos
        
        # Update Kalman filter
        noise = np.random.normal(0, 2.0, 2)
        self.kalman.update(self.position + noise)
        self.kalman.predict()
        
        # Update connection
        old_bs = self.connected_bs
        self._update_connection()
        
        if self.connected_bs == old_bs and self.connected_bs is not None:
            self.time_in_current_bs += dt
            
        # Consume resources
        self.resource_level = max(0.1, self.resource_level - 0.0005 * dt)
    
    def simulate_round_movement(self):
        """Simulate vehicle movement for one FL round duration"""
        steps = int(self.config.round_duration / self.config.position_update_interval)
        for _ in range(steps):
            self.update_position(self.config.position_update_interval)
    
    def determine_training_outcome(self) -> ClientState:
        """
        Determine training outcome based on ACTUAL physical conditions.
        This is the same for all methods - physics determines outcomes.
        """
        rssi = self.get_rssi()
        rssi_norm = self.get_normalized_rssi()
        contact_time = self.get_contact_time()
        speed = np.linalg.norm(self.velocity)
        
        # 1. Connection failure: based on actual RSSI
        if rssi < self.config.rssi_min_connection:
            return ClientState.CONNECTION_FAILURE
        
        # Higher chance of connection failure with poor signal
        conn_fail_prob = max(0.05, 0.35 * (1 - rssi_norm))
        
        # Recently handed over = higher failure chance
        if self.time_in_current_bs < 5.0:
            conn_fail_prob += 0.15
            
        if random.random() < conn_fail_prob:
            return ClientState.CONNECTION_FAILURE
        
        # 2. Resource failure: based on resource level
        resource_fail_prob = max(0.05, 0.4 * (1 - self.resource_level))
        if random.random() < resource_fail_prob:
            return ClientState.RESOURCE_FAILURE
        
        # 3. Abort: based on contact time
        if contact_time < self.config.round_duration:
            abort_prob = 0.3 + 0.5 * (1 - contact_time / self.config.round_duration)
            if random.random() < abort_prob:
                return ClientState.ABORT
        
        # Also abort probability from high speed
        speed_factor = min(1.0, speed / 25.0)
        if random.random() < 0.1 * speed_factor:
            return ClientState.ABORT
        
        return ClientState.SUCCESS
    
    def record_outcome(self, outcome: ClientState):
        """Record outcome in history"""
        self.outcome_history.append(outcome)
        
        if outcome != ClientState.SUCCESS:
            self.consecutive_failures += 1
            self.total_dropouts += 1
        else:
            self.consecutive_failures = 0
    
    def record_accuracy(self, accuracy: float):
        """Record local training accuracy"""
        self.accuracy_history.append(accuracy)
    
    def get_entropy(self) -> float:
        """Get data entropy (diversity)"""
        return self.base_entropy
    
    def get_average_accuracy(self, window: int = 10) -> float:
        """Get average local accuracy from recent rounds"""
        if not self.accuracy_history:
            return 0.5  # Default
        recent = self.accuracy_history[-window:]
        return np.mean(recent)
    
    def get_success_rate(self, window: int = 10) -> float:
        """Get recent success rate"""
        if not self.outcome_history:
            return 0.7  # Default
        recent = self.outcome_history[-window:]
        return sum(1 for s in recent if s == ClientState.SUCCESS) / len(recent)
    
    @property
    def num_samples(self) -> int:
        return len(self.data_indices)


def create_clients(config: ExperimentConfig, 
                   base_stations: List[BaseStation],
                   client_data_indices: List[List[int]]) -> List[VehicularClient]:
    """Create all vehicular clients"""
    clients = []
    for i in range(config.num_clients):
        client = VehicularClient(
            client_id=i,
            config=config,
            base_stations=base_stations,
            data_indices=client_data_indices[i]
        )
        clients.append(client)
    return clients
