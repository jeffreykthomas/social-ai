# Predictive Agents Implementation Guide

## Overview

This document describes the implementation of the need-based predictive agent architecture for the AI Playground. The system creates agents that continuously predict future events and their impact on internal need states, maintaining a 24/7 internal monologue that occasionally surfaces as external communication.

## Architecture Components

### 1. Core Agent System (`predictive_persona.py`)

The main agent class that implements:
- **Need State Management**: Tracks 9 core needs (connection, safety, approval, empathy, fun, attention, achievement, autonomy, purpose)
- **Dual Prediction System**: Predicts both events and their impact on needs
- **Continuous Internal Monologue**: Maintains stream of consciousness
- **Learning from Prediction Errors**: Updates models based on actual vs predicted outcomes

Key classes:
- `PredictivePersona`: Main agent class
- `NeedState`: Manages need values and decay
- `PredictionModel`: Dual prediction for events and impacts
- `InternalMonologue`: Thought stream management

### 2. LLM Integration (`need_aware_prompts.py`)

Sophisticated prompt engineering that:
- Incorporates need states into natural language generation
- Validates responses for consistency with agent's psychological state
- Provides multiple prompt templates for different scenarios
- Ensures agents don't explicitly mention their internal mechanics

### 3. Visualization System (`agent_monitor.html`)

Real-time monitoring interface showing:
- Need state bars with color coding
- Internal monologue stream
- Current predictions with probabilities
- Social models of other agents
- Interactive controls for educational demonstrations

### 4. Configuration System (`need_config.yaml`)

Comprehensive configuration for:
- Need definitions and parameters
- Learning rates and thresholds
- Event impact patterns
- Social interaction weights
- Performance optimization settings

### 5. Integration Layer (`persona_manager.py`)

Manages:
- Agent lifecycle and updates
- Inter-agent interactions
- WebSocket connections for real-time monitoring
- Event processing and distribution
- Compatibility with original Reverie system

## Key Features

### Need-Based Decision Making

Agents make decisions by:
1. Continuously monitoring need states
2. Predicting how events will impact needs
3. Selecting actions that maximize expected need fulfillment
4. Learning from prediction errors

### Internal Monologue

The continuous thought stream includes:
- Need awareness thoughts
- Event predictions
- Social modeling
- Decision reasoning
- Observations

### Social Modeling

Agents maintain models of other agents':
- Estimated need states
- Likely behaviors
- Interaction history
- Mutual benefit calculations

### Educational Features

The system supports learning through:
- Transparent need reasoning
- Prediction visualization
- Learning progress tracking
- Manual intervention capabilities
- Detailed explanations

## Usage Guide

### Starting the System

1. Configure needs in `need_config.yaml`
2. Start Django server: `python manage.py runserver`
3. Start simulation: `python reverie.py`
4. Access monitor at: `http://localhost:8000/agent_monitor`

### Creating Agents

```python
from persona_manager import get_manager

manager = get_manager(use_predictive=True)
agent = manager.create_agent("Isabella Rodriguez")
```

### Running Simulation

```python
import asyncio

async def run():
    manager = get_manager()
    while True:
        await manager.update_agents()
        await asyncio.sleep(1)  # 1 second timesteps

asyncio.run(run())
```

### Monitoring Agents

The web interface provides:
- Real-time need state visualization
- Internal monologue display
- Prediction tracking
- Learning metrics

### Educational Demonstrations

Use the API endpoints to:
- Manually adjust need states
- Trigger specific events
- Pause/resume agents
- Get detailed explanations

## API Endpoints

- `GET /agent_monitor/` - Main monitoring interface
- `GET /api/agent_states/` - Current states of all agents
- `GET /api/agent/<name>/` - Detailed info for specific agent
- `POST /api/agent/<name>/need/` - Update agent need
- `POST /api/trigger_event/` - Trigger environmental event
- `GET /api/agent/<name>/predictions/` - Get prediction explanations
- `GET /api/agent/<name>/learning/` - Get learning progress

## Configuration Options

### Need Parameters
- `initial_value`: Starting fulfillment level (0-1)
- `decay_rate`: How quickly need depletes per timestep
- `priority_weight`: Importance in decision making
- `critical_threshold`: When need becomes urgent

### Learning Parameters
- `prediction_learning_rate`: How quickly predictions improve
- `exploration_rate`: Balance between exploitation/exploration
- `social_learning_rate`: How quickly social models update

### Performance Settings
- `enable_parallel_predictions`: Run agents in parallel
- `cache_predictions`: Cache recent predictions
- `max_computation_time`: Time limit per think cycle

## Integration with Original System

The predictive agents can run alongside original Reverie agents:
- Use `use_predictive=False` for original behavior
- Both types can exist in same simulation
- Shared spatial and memory systems
- Compatible event handling

## Future Enhancements

1. **Advanced Learning**
   - Neural network prediction models
   - Transfer learning between agents
   - Meta-learning capabilities

2. **Richer Interactions**
   - Multi-step planning
   - Coalition formation
   - Emotional contagion

3. **Educational Tools**
   - Guided tutorials
   - Scenario editor
   - Comparative analysis tools

## Troubleshooting

### Common Issues

1. **High CPU Usage**
   - Reduce number of agents
   - Increase think cycle interval
   - Disable parallel processing

2. **Prediction Errors**
   - Check event definitions
   - Verify need impact patterns
   - Increase learning rate

3. **WebSocket Disconnections**
   - Check Django Channels setup
   - Verify ASGI configuration
   - Monitor network stability

### Debug Mode

Enable debug output:
```python
# In need_config.yaml
debug: true
```

This provides detailed logging of:
- Prediction calculations
- Need updates
- Decision processes
- Learning updates

## Conclusion

The predictive need-based agent system provides a sophisticated framework for creating psychologically realistic agents that learn and adapt through experience. The combination of continuous internal reasoning, dual prediction systems, and transparent monitoring makes it ideal for educational applications in social skill development.