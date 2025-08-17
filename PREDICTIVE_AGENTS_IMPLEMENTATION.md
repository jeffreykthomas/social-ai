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
- **Time-Budgeting**: Chooses time budgets per task, with re-evaluation checkpoints

Key classes:
- `PredictivePersona`: Main agent class
- `NeedState`: Manages need values and decay
- `PredictionModel`: Dual prediction for events and impacts
- `InternalMonologue`: Thought stream management
- `TimeAllocator`: Computes time budgets and switching decisions

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
- Reputation indicators and social notes per agent-pair
- Time budgets and upcoming schedule

### 4. Configuration System (`need_config.yaml`)

Comprehensive configuration for:
- Need definitions and parameters
- Learning rates and thresholds
- Event impact patterns
- Social interaction weights
- Performance optimization settings
- Time-allocation settings (evaluation interval, switch penalty, min-session length)

### 5. Integration Layer (`persona_manager.py`)

Manages:
- Agent lifecycle and updates
- Inter-agent interactions
- WebSocket connections for real-time monitoring
- Event processing and distribution
- Compatibility with original Reverie system

#### Interaction Initiation Flow

1. Candidate selection: find partners with required complementary tools/capabilities.
2. Inquiry: send capability request/offer including proposed time budget and scope.
3. Negotiation: adjust terms to maximize mutual expected fulfillment.
4. Commit: create an interaction session with entry/exit conditions.
5. Execute: run tool calls (one side per agent) and monitor progress.
6. Evaluate: perform post-interaction evaluation and update social notes.

Abrupt exits are recorded by the counterparty and reduce future selection probability.
- Partner discovery and interaction initiation

## Key Features

### Need-Based Decision Making

Agents make decisions by:
1. Continuously monitoring need states
2. Predicting how events will impact needs
3. Selecting actions that maximize expected need fulfillment
4. Learning from prediction errors

### Time Allocation Predictions

Agents predict not only which action to take but also how much time to allocate to it. They estimate value-density (expected need fulfillment per unit time) and compare it against alternatives and schedule constraints. If marginal value falls below the best available alternative, agents plan to exit at the next safe breakpoint (e.g., after a transaction completes).

### Internal Monologue

The continuous thought stream includes:
- Need awareness thoughts
- Event predictions
- Social modeling
- Decision reasoning
- Observations
- Post-interaction evaluation and time-allocation adjustments

### Social Modeling

Agents maintain models of other agents':
- Estimated need states
- Likely behaviors
- Interaction history
- Mutual benefit calculations
- Tool capabilities and reliability (including propensity to leave early)

### Educational Features

The system supports learning through:
- Transparent need reasoning
- Prediction visualization
- Learning progress tracking
- Manual intervention capabilities
- Detailed explanations

## Usage Guide

### Schedules and Tasks

- Define each agent’s daily schedule as a list of tasks with optional windows, priorities, and target needs.
- Tasks may require tools or partner capabilities; specify `requires: [tool_names]` and `counterparty: role` when applicable.
- Agents re-plan opportunistically when higher-value options become available.

### Tools and Transactions

- Tools represent one side of a transaction (e.g., `purchase_groceries` vs `sell_groceries`).
- Each agent has a declared `tool_abilities` set; agents can communicate capabilities and inquire about others.
- Partner discovery finds complementary counterparts for required tools.
- When a task requires a counterparty, both agents execute complementary tool calls; success emits a `transaction_complete` event.

### Interaction Lifecycle and Exit Behavior

1. Initiate: capability inquiry/offer with proposed time budget.
2. Negotiate: adjust scope/time to maximize mutual expected fulfillment.
3. Execute: run complementary tools while monitoring marginal value.
4. Evaluate: short evaluation period to log outcomes and update notes.
5. Continue or Exit: continue if value-density remains high; otherwise exit at the next safe breakpoint.

- Agents may exit at any time. Exiting before a safe breakpoint is considered abrupt and results in the counterparty recording a note such as "may leave unpredictably", decreasing future selection probability. The magnitude is controlled by `abrupt_exit_penalty`.

### Post-Interaction Evaluation

- Compare predicted vs actual fulfillment and time used.
- Update social notes (e.g., capabilities, empathy, reliability).
- Adjust partner trust and future time budgets.
- Decide next task from schedule based on updated predictions.

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

# Example: record a social note about another agent
manager.add_note(observer="Isabella Rodriguez", about="Bill", note="can sell groceries; very giving of emotional support")
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
- Schedule view with time budgets and next re-evaluation
- Social notes and reputation summaries

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
- `GET /api/agent/<name>/schedule/` - Current schedule and time budgets
- `POST /api/agent/<name>/notes/` - Append or update social notes on another agent
- `POST /api/interaction/initiate/` - Initiate an interaction with required tools/capabilities
- `POST /api/interaction/exit/` - Exit an interaction (reason and abruptness flags)

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

### Time Allocation Settings
- `evaluation_interval_steps`: How often to reassess time allocation
- `min_session_steps`: Minimum time before considering a switch
- `switch_penalty`: Cost applied when switching tasks or partners
- `abrupt_exit_penalty`: Reputation impact recorded by partners on abrupt exit
- `post_interaction_eval_steps`: Time reserved for evaluation after an interaction

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
   - Partner choice modeling with reputational dynamics
   - Market-style matching for complementary tools (buyer/seller)

3. **Educational Tools**
   - Guided tutorials
   - Scenario editor
   - Comparative analysis tools
   - Replay annotator for social notes and exit decisions

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