"""
Predictive Persona with Need-Based Architecture

This module implements agents that operate through predictive processing 
centered on need fulfillment. Each agent maintains:
- Continuous internal monologue about future events and need states
- Dual prediction system for events and their need impacts
- Learning from prediction errors
- Social modeling of other agents' needs

Author: AI Playground Team
"""

import numpy as np
import json
import datetime
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import sys
sys.path.append('../')

from global_methods import *
from persona.memory_structures.spatial_memory import *
from persona.memory_structures.associative_memory import *
from persona.memory_structures.scratch import *
from persona.cognitive_modules.perceive import *
from persona.cognitive_modules.retrieve import *


class NeedState:
    """Manages an agent's internal need states"""
    
    def __init__(self):
        # Core needs with initial values
        self.needs = {
            'connection': 0.5,
            'safety': 0.8,
            'approval': 0.6,
            'empathy': 0.4,
            'fun': 0.3,
            'attention': 0.5,
            'achievement': 0.5,
            'autonomy': 0.7,
            'purpose': 0.6
        }
        
        # Decay rates for each need (per timestep)
        self.decay_rates = {
            'connection': 0.01,
            'safety': 0.005,
            'approval': 0.008,
            'empathy': 0.012,
            'fun': 0.015,
            'attention': 0.01,
            'achievement': 0.008,
            'autonomy': 0.006,
            'purpose': 0.004
        }
        
        # History for tracking changes
        self.history = []
        
    def decay(self):
        """Apply natural decay to all needs"""
        for need, rate in self.decay_rates.items():
            self.needs[need] = max(0, self.needs[need] - rate)
        self._record_state()
    
    def update(self, need: str, change: float):
        """Update a specific need value"""
        if need in self.needs:
            old_value = self.needs[need]
            self.needs[need] = np.clip(self.needs[need] + change, 0, 1)
            return self.needs[need] - old_value
        return 0
    
    def get_deficiency_scores(self) -> Dict[str, float]:
        """Calculate how deficient each need is (1 - fulfillment)"""
        return {need: 1 - value for need, value in self.needs.items()}
    
    def get_most_deficient(self, n: int = 3) -> List[str]:
        """Get the n most deficient needs"""
        deficiencies = self.get_deficiency_scores()
        sorted_needs = sorted(deficiencies.items(), key=lambda x: x[1], reverse=True)
        return [need for need, _ in sorted_needs[:n]]
    
    def _record_state(self):
        """Record current state in history"""
        self.history.append({
            'timestamp': datetime.datetime.now(),
            'needs': self.needs.copy()
        })
        # Keep only last 1000 entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'needs': self.needs,
            'decay_rates': self.decay_rates
        }


class PredictionModel:
    """Dual prediction model for events and need impacts"""
    
    def __init__(self):
        # Event prediction components
        self.event_vocabulary = set()
        self.event_sequences = deque(maxlen=1000)
        self.event_transition_counts = {}
        
        # Need impact model
        self.event_need_impacts = {}  # event -> {need: (mean, std)}
        self.context_modifiers = {}   # context features that modify impacts
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        
    def predict_next_events(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict probability distribution over next events"""
        recent_events = context.get('recent_events', [])
        if not recent_events:
            # Uniform distribution if no history
            if self.event_vocabulary:
                prob = 1.0 / len(self.event_vocabulary)
                return {event: prob for event in self.event_vocabulary}
            return {}
        
        # Use event transition model
        last_event = recent_events[-1]
        if last_event in self.event_transition_counts:
            transitions = self.event_transition_counts[last_event]
            total = sum(transitions.values())
            if total > 0:
                return {event: count/total for event, count in transitions.items()}
        
        # Fallback to uniform
        if self.event_vocabulary:
            prob = 1.0 / len(self.event_vocabulary)
            return {event: prob for event in self.event_vocabulary}
        return {}
    
    def predict_need_impact(self, event: str, current_needs: Dict[str, float]) -> Dict[str, float]:
        """Predict how an event will impact needs"""
        impacts = {}
        
        if event in self.event_need_impacts:
            for need, (mean, std) in self.event_need_impacts[event].items():
                # Sample from learned distribution
                impact = np.random.normal(mean, std)
                # Modulate by current need state (less impact when already fulfilled)
                current_level = current_needs.get(need, 0.5)
                impact *= (1 - current_level) if impact > 0 else current_level
                impacts[need] = impact
        else:
            # Unknown event - small random impacts
            for need in current_needs:
                impacts[need] = np.random.normal(0, 0.05)
        
        return impacts
    
    def update_from_experience(self, event: str, actual_impacts: Dict[str, float], 
                              predicted_impacts: Dict[str, float]):
        """Update model based on prediction error"""
        # Update event vocabulary
        self.event_vocabulary.add(event)
        
        # Update impact model
        if event not in self.event_need_impacts:
            self.event_need_impacts[event] = {}
        
        for need, actual_impact in actual_impacts.items():
            predicted = predicted_impacts.get(need, 0)
            error = actual_impact - predicted
            
            if need not in self.event_need_impacts[event]:
                # Initialize with observed impact
                self.event_need_impacts[event][need] = (actual_impact, 0.1)
            else:
                # Update mean and std using online learning
                old_mean, old_std = self.event_need_impacts[event][need]
                new_mean = old_mean + self.learning_rate * error
                new_std = old_std * 0.95 + 0.05 * abs(error)  # Decay std, add error
                self.event_need_impacts[event][need] = (new_mean, new_std)
    
    def update_event_sequence(self, prev_event: str, next_event: str):
        """Update event transition model"""
        if prev_event not in self.event_transition_counts:
            self.event_transition_counts[prev_event] = {}
        
        if next_event not in self.event_transition_counts[prev_event]:
            self.event_transition_counts[prev_event][next_event] = 0
        
        self.event_transition_counts[prev_event][next_event] += 1
        self.event_sequences.append((prev_event, next_event))


class InternalMonologue:
    """Manages the continuous internal monologue stream"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.thoughts = deque(maxlen=500)  # Keep last 500 thoughts
        self.thought_patterns = {
            'need_awareness': [
                "My {need} is getting low ({value:.2f})",
                "I'm feeling the lack of {need}",
                "My {need} need is quite fulfilled ({value:.2f})",
                "I should focus on improving my {need}"
            ],
            'prediction': [
                "I predict {event} will happen soon",
                "If {event} occurs, my {need} will likely {direction}",
                "{event} might lead to {outcome}",
                "I expect {probability:.0%} chance of {event}"
            ],
            'social_modeling': [
                "I think {agent} needs {need} based on their {behavior}",
                "{agent} seems to be seeking {need}",
                "If I {action}, {agent} will probably {response}",
                "{agent}'s {need} appears low"
            ],
            'decision': [
                "I should {action} to improve my {need}",
                "{action} seems like the best way to fulfill my {need}",
                "The risk of {negative} is worth the potential {positive}",
                "I'll try {action} since my {need} is critical"
            ]
        }
        
    def add_thought(self, thought_type: str, **kwargs):
        """Add a structured thought to the monologue"""
        if thought_type in self.thought_patterns:
            patterns = self.thought_patterns[thought_type]
            pattern = random.choice(patterns)
            thought = pattern.format(**kwargs)
        else:
            thought = kwargs.get('content', '*thinking*')
        
        timestamp = datetime.datetime.now()
        self.thoughts.append({
            'timestamp': timestamp,
            'type': thought_type,
            'content': thought,
            'metadata': kwargs
        })
        
        return thought
    
    def get_recent_thoughts(self, n: int = 50) -> List[str]:
        """Get the n most recent thoughts as strings"""
        return [t['content'] for t in list(self.thoughts)[-n:]]
    
    def get_thoughts_by_type(self, thought_type: str, n: int = 10) -> List[Dict]:
        """Get recent thoughts of a specific type"""
        typed_thoughts = [t for t in self.thoughts if t['type'] == thought_type]
        return list(typed_thoughts)[-n:]
    
    def generate_summary(self) -> str:
        """Generate a summary of recent thinking"""
        recent = list(self.thoughts)[-20:]
        if not recent:
            return "Just starting to think..."
        
        # Count thought types
        type_counts = {}
        for thought in recent:
            type_counts[thought['type']] = type_counts.get(thought['type'], 0) + 1
        
        # Find dominant concerns
        need_mentions = {}
        for thought in recent:
            if 'need' in thought['metadata']:
                need = thought['metadata']['need']
                need_mentions[need] = need_mentions.get(need, 0) + 1
        
        summary_parts = []
        if need_mentions:
            top_need = max(need_mentions.items(), key=lambda x: x[1])[0]
            summary_parts.append(f"Mostly thinking about {top_need}")
        
        if 'prediction' in type_counts:
            summary_parts.append(f"Making {type_counts['prediction']} predictions")
        
        return ". ".join(summary_parts) if summary_parts else "Processing..."


class PredictivePersona:
    """
    Predictive agent with need-based architecture and continuous internal monologue
    """
    
    def __init__(self, name: str, folder_mem_saved: Optional[str] = None):
        # Basic identity
        self.name = name
        self.agent_id = f"agent_{name.lower().replace(' ', '_')}"
        
        # Need management
        self.needs = NeedState()
        
        # Prediction systems
        self.prediction_model = PredictionModel()
        self.prediction_buffer = deque(maxlen=100)  # Recent predictions
        self.prediction_errors = deque(maxlen=100)  # Recent errors
        
        # Internal monologue
        self.monologue = InternalMonologue(name)
        
        # Memory systems (inherited from original)
        if folder_mem_saved:
            f_s_mem_saved = f"{folder_mem_saved}/bootstrap_memory/spatial_memory.json"
            self.s_mem = MemoryTree(f_s_mem_saved)
            f_a_mem_saved = f"{folder_mem_saved}/bootstrap_memory/associative_memory"
            self.a_mem = AssociativeMemory(f_a_mem_saved)
            scratch_saved = f"{folder_mem_saved}/bootstrap_memory/scratch.json"
            self.scratch = Scratch(scratch_saved)
        else:
            self.s_mem = MemoryTree()
            self.a_mem = AssociativeMemory()
            self.scratch = Scratch()
        
        # Social modeling
        self.other_agent_models = {}  # agent_id -> PredictionModel
        
        # Action planning
        self.current_plan = []
        self.action_history = deque(maxlen=100)
        
        # Environmental awareness
        self.current_location = None
        self.nearby_agents = []
        self.recent_events = deque(maxlen=50)
        
        # Initialize thinking
        self.monologue.add_thought('need_awareness', 
                                  need='safety', 
                                  value=self.needs.needs['safety'])
    
    def think_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Main thinking loop - continuous prediction and need monitoring
        Returns action if one should be taken
        """
        # Decay needs
        self.needs.decay()
        
        # Check need states and add thoughts
        deficient_needs = self.needs.get_most_deficient(2)
        for need in deficient_needs:
            if self.needs.needs[need] < 0.3:  # Critical threshold
                self.monologue.add_thought('need_awareness',
                                         need=need,
                                         value=self.needs.needs[need])
        
        # Generate predictions
        context = self._build_prediction_context()
        event_predictions = self.prediction_model.predict_next_events(context)
        
        # Evaluate each possible event's impact on needs
        best_outcome_score = -float('inf')
        best_action = None
        
        for event, probability in event_predictions.items():
            if probability < 0.1:  # Skip unlikely events
                continue
                
            # Predict need impacts
            predicted_impacts = self.prediction_model.predict_need_impact(
                event, self.needs.needs
            )
            
            # Calculate expected value
            future_needs = self.needs.needs.copy()
            for need, impact in predicted_impacts.items():
                future_needs[need] = np.clip(future_needs[need] + impact, 0, 1)
            
            score = self._calculate_need_fulfillment_score(future_needs)
            weighted_score = score * probability
            
            # Add prediction thought
            primary_impact = max(predicted_impacts.items(), key=lambda x: abs(x[1]))
            self.monologue.add_thought('prediction',
                                     event=event,
                                     need=primary_impact[0],
                                     direction='increase' if primary_impact[1] > 0 else 'decrease',
                                     probability=probability)
            
            # Track if this is the best predicted outcome
            if weighted_score > best_outcome_score:
                best_outcome_score = weighted_score
                best_action = self._determine_action_for_event(event)
        
        # Decide whether to act
        current_score = self._calculate_need_fulfillment_score(self.needs.needs)
        improvement = best_outcome_score - current_score
        
        if improvement > 0.1 and best_action:  # Significant improvement threshold
            self.monologue.add_thought('decision',
                                     action=best_action['type'],
                                     need=deficient_needs[0])
            return best_action
        
        return None
    
    def observe_event(self, event: Dict[str, Any]):
        """Process an observed event and update predictions"""
        event_type = event.get('type', 'unknown')
        self.recent_events.append(event)
        
        # Calculate actual need impacts
        actual_impacts = self._calculate_event_impact(event)
        
        # Retrieve prediction if we have one
        predicted_impacts = {}
        for pred in self.prediction_buffer:
            if pred['event'] == event_type:
                predicted_impacts = pred['predicted_impacts']
                break
        
        # Update needs
        for need, impact in actual_impacts.items():
            self.needs.update(need, impact)
        
        # Learn from prediction error
        if predicted_impacts:
            self.prediction_model.update_from_experience(
                event_type, actual_impacts, predicted_impacts
            )
            
            # Record error
            error = sum(abs(actual_impacts.get(n, 0) - predicted_impacts.get(n, 0)) 
                       for n in self.needs.needs)
            self.prediction_errors.append({
                'event': event_type,
                'error': error,
                'timestamp': datetime.datetime.now()
            })
        
        # Update internal monologue
        impact_summary = self._summarize_impact(actual_impacts)
        self.monologue.add_thought('observation',
                                 content=f"*{event_type} happened - {impact_summary}*")
    
    def interact_with_agent(self, other_agent: 'PredictivePersona', 
                           interaction_type: str) -> Dict[str, Any]:
        """Handle interaction with another agent"""
        # Model their needs if not already done
        if other_agent.agent_id not in self.other_agent_models:
            self.other_agent_models[other_agent.agent_id] = {
                'name': other_agent.name,
                'estimated_needs': self._estimate_other_needs(other_agent),
                'interaction_history': []
            }
        
        # Add social modeling thought
        other_model = self.other_agent_models[other_agent.agent_id]
        primary_need = max(other_model['estimated_needs'].items(), 
                          key=lambda x: 1 - x[1])[0]
        
        self.monologue.add_thought('social_modeling',
                                 agent=other_agent.name,
                                 need=primary_need,
                                 behavior=interaction_type)
        
        # Generate response based on mutual need fulfillment
        response = self._generate_social_response(other_agent, interaction_type)
        
        # Update interaction history
        other_model['interaction_history'].append({
            'type': interaction_type,
            'response': response,
            'timestamp': datetime.datetime.now()
        })
        
        return response
    
    def externalize_thought(self) -> Optional[str]:
        """
        Convert internal monologue to external speech when appropriate
        """
        recent_thoughts = self.monologue.get_recent_thoughts(10)
        if not recent_thoughts:
            return None
        
        # Check if we should speak (based on social context and needs)
        if self.needs.needs['connection'] < 0.4 and self.nearby_agents:
            # Need connection and others are present
            context = {
                'internal_monologue': ' '.join(recent_thoughts),
                'current_needs': self.needs.to_dict(),
                'nearby_agents': [a.name for a in self.nearby_agents],
                'recent_events': list(self.recent_events)[-5:]
            }
            
            # This is where we'd integrate with LLM
            # For now, return a simplified version
            thought_summary = self.monologue.generate_summary()
            return f"[{self.name} thinking about {thought_summary}]"
        
        return None
    
    # Helper methods
    
    def _build_prediction_context(self) -> Dict[str, Any]:
        """Build context for prediction model"""
        return {
            'current_needs': self.needs.needs,
            'recent_thoughts': self.monologue.get_recent_thoughts(20),
            'recent_events': [e.get('type', 'unknown') for e in self.recent_events],
            'location': self.current_location,
            'nearby_agents': [a.agent_id for a in self.nearby_agents],
            'time': datetime.datetime.now()
        }
    
    def _calculate_need_fulfillment_score(self, need_state: Dict[str, float]) -> float:
        """Calculate overall fulfillment score with priority weighting"""
        # Weight critical needs more heavily
        weights = {
            'safety': 2.0,
            'connection': 1.5,
            'approval': 1.2,
            'empathy': 1.0,
            'fun': 0.8,
            'attention': 1.0,
            'achievement': 1.1,
            'autonomy': 1.3,
            'purpose': 1.4
        }
        
        weighted_sum = sum(need_state[need] * weights[need] for need in need_state)
        total_weight = sum(weights.values())
        
        return weighted_sum / total_weight
    
    def _determine_action_for_event(self, event: str) -> Optional[Dict[str, Any]]:
        """Determine what action would lead to the predicted event"""
        # This would be more sophisticated in full implementation
        action_map = {
            'conversation_starts': {'type': 'initiate_conversation'},
            'receive_help': {'type': 'ask_for_help'},
            'give_help': {'type': 'offer_help'},
            'join_activity': {'type': 'approach_group'},
            'solo_activity': {'type': 'find_quiet_space'}
        }
        
        return action_map.get(event)
    
    def _calculate_event_impact(self, event: Dict[str, Any]) -> Dict[str, float]:
        """Calculate actual impact of an event on needs"""
        event_type = event.get('type', 'unknown')
        impacts = {}
        
        # Define impact patterns
        impact_patterns = {
            'conversation_starts': {
                'connection': 0.3,
                'attention': 0.2,
                'fun': 0.1
            },
            'conversation_ends': {
                'connection': -0.1,
                'autonomy': 0.1
            },
            'receive_compliment': {
                'approval': 0.4,
                'connection': 0.1
            },
            'give_help': {
                'purpose': 0.3,
                'connection': 0.2,
                'empathy': 0.2
            }
        }
        
        if event_type in impact_patterns:
            impacts = impact_patterns[event_type]
        else:
            # Small random impacts for unknown events
            for need in self.needs.needs:
                impacts[need] = np.random.normal(0, 0.05)
        
        return impacts
    
    def _summarize_impact(self, impacts: Dict[str, float]) -> str:
        """Create human-readable summary of need impacts"""
        significant = [(n, v) for n, v in impacts.items() if abs(v) > 0.1]
        if not significant:
            return "minimal impact"
        
        parts = []
        for need, value in sorted(significant, key=lambda x: abs(x[1]), reverse=True)[:2]:
            direction = "increased" if value > 0 else "decreased"
            parts.append(f"{need} {direction}")
        
        return ", ".join(parts)
    
    def _estimate_other_needs(self, other_agent: 'PredictivePersona') -> Dict[str, float]:
        """Estimate another agent's need states based on observations"""
        # Simple heuristic - would be learned in full implementation
        estimated = {}
        for need in self.needs.needs:
            # Start with assumption of moderate fulfillment
            estimated[need] = 0.5 + np.random.normal(0, 0.1)
        
        return estimated
    
    def _generate_social_response(self, other_agent: 'PredictivePersona', 
                                 interaction_type: str) -> Dict[str, Any]:
        """Generate response based on mutual need fulfillment prediction"""
        # Simplified version - would use LLM in full implementation
        responses = {
            'greeting': {
                'type': 'greeting_response',
                'content': f"Hello {other_agent.name}!",
                'predicted_impact': {'connection': 0.1}
            },
            'help_request': {
                'type': 'help_response',
                'content': "I'd be happy to help!",
                'predicted_impact': {'purpose': 0.2, 'connection': 0.2}
            }
        }
        
        return responses.get(interaction_type, {
            'type': 'generic_response',
            'content': "I see...",
            'predicted_impact': {}
        })
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            'name': self.name,
            'needs': self.needs.to_dict(),
            'recent_thoughts': list(self.monologue.thoughts),
            'prediction_errors': list(self.prediction_errors),
            'other_agent_models': self.other_agent_models
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load agent state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.name = state['name']
        self.needs.needs = state['needs']['needs']
        self.needs.decay_rates = state['needs']['decay_rates']
        # Restore other state as needed