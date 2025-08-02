"""
Configuration Loader for Need-Based Predictive Agents

This module handles loading and managing configuration from YAML files,
providing easy access to need parameters, learning rates, and other settings.

Author: AI Playground Team
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class NeedConfig:
    """Manages configuration for the need-based agent system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to config file in same directory
            config_path = Path(__file__).parent / "need_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()
    
    # Need configuration accessors
    
    def get_need_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get all need definitions"""
        return self.config.get('needs', {})
    
    def get_need_info(self, need_name: str) -> Dict[str, Any]:
        """Get information for a specific need"""
        return self.config['needs'].get(need_name, {})
    
    def get_initial_needs(self) -> Dict[str, float]:
        """Get initial values for all needs"""
        return {
            need: info['initial_value'] 
            for need, info in self.config['needs'].items()
        }
    
    def get_decay_rates(self) -> Dict[str, float]:
        """Get decay rates for all needs"""
        return {
            need: info['decay_rate'] 
            for need, info in self.config['needs'].items()
        }
    
    def get_priority_weights(self) -> Dict[str, float]:
        """Get priority weights for needs"""
        return {
            need: info['priority_weight'] 
            for need, info in self.config['needs'].items()
        }
    
    def get_critical_thresholds(self) -> Dict[str, float]:
        """Get critical thresholds for needs"""
        return {
            need: info['critical_threshold'] 
            for need, info in self.config['needs'].items()
        }
    
    # Learning parameters
    
    def get_learning_params(self) -> Dict[str, Any]:
        """Get all learning parameters"""
        return self.config.get('learning', {})
    
    def get_prediction_learning_rate(self) -> float:
        """Get prediction model learning rate"""
        return self.config['learning']['prediction_learning_rate']
    
    def get_exploration_rate(self) -> float:
        """Get exploration rate for predictions"""
        return self.config['learning']['exploration_rate']
    
    # Action selection
    
    def get_action_params(self) -> Dict[str, Any]:
        """Get action selection parameters"""
        return self.config.get('action_selection', {})
    
    def get_action_threshold(self) -> float:
        """Get minimum improvement needed to take action"""
        return self.config['action_selection']['action_threshold']
    
    # Internal monologue
    
    def get_monologue_params(self) -> Dict[str, Any]:
        """Get internal monologue parameters"""
        return self.config.get('monologue', {})
    
    def get_thought_type_weights(self) -> Dict[str, float]:
        """Get weights for different thought types"""
        return self.config['monologue']['thought_type_weights']
    
    def get_externalization_threshold(self) -> float:
        """Get connection need threshold for externalization"""
        return self.config['monologue']['externalization_threshold']
    
    # Event impacts
    
    def get_event_impacts(self) -> Dict[str, Dict[str, float]]:
        """Get all event impact patterns"""
        return self.config.get('event_impacts', {})
    
    def get_event_impact(self, event_type: str) -> Dict[str, float]:
        """Get impact pattern for a specific event"""
        return self.config['event_impacts'].get(event_type, {})
    
    # Social parameters
    
    def get_social_params(self) -> Dict[str, Any]:
        """Get social interaction parameters"""
        return self.config.get('social', {})
    
    # Monitoring and visualization
    
    def get_monitoring_params(self) -> Dict[str, Any]:
        """Get monitoring parameters"""
        return self.config.get('monitoring', {})
    
    def get_ui_update_interval(self) -> float:
        """Get UI update interval in seconds"""
        return self.config['monitoring']['ui_update_interval']
    
    # Educational features
    
    def get_education_params(self) -> Dict[str, Any]:
        """Get educational feature parameters"""
        return self.config.get('education', {})
    
    def should_generate_explanations(self) -> bool:
        """Check if explanations should be generated"""
        return self.config['education']['generate_explanations']
    
    # Performance
    
    def get_performance_params(self) -> Dict[str, Any]:
        """Get performance optimization parameters"""
        return self.config.get('performance', {})
    
    def is_parallel_enabled(self) -> bool:
        """Check if parallel predictions are enabled"""
        return self.config['performance']['enable_parallel_predictions']
    
    # Utility methods
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self.config.copy()
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)


# Global configuration instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> NeedConfig:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to config file (only used on first call)
    
    Returns:
        NeedConfig instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = NeedConfig(config_path)
    
    return _config_instance


def reset_config():
    """Reset global configuration instance"""
    global _config_instance
    _config_instance = None


# Convenience functions for common access patterns

def get_initial_needs() -> Dict[str, float]:
    """Get initial need values"""
    return get_config().get_initial_needs()


def get_decay_rates() -> Dict[str, float]:
    """Get need decay rates"""
    return get_config().get_decay_rates()


def get_event_impact(event_type: str) -> Dict[str, float]:
    """Get impact of an event on needs"""
    return get_config().get_event_impact(event_type)


def get_learning_rate() -> float:
    """Get prediction learning rate"""
    return get_config().get_prediction_learning_rate()


def get_action_threshold() -> float:
    """Get action selection threshold"""
    return get_config().get_action_threshold()