"""
Persona Manager for Predictive Need-Based Agents

This module manages the integration of predictive personas with the existing
Reverie system, handling agent creation, updates, and interactions.

Author: AI Playground Team
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from predictive_persona import PredictivePersona
from persona import Persona  # Original persona class
from cognitive_modules.need_aware_prompts import create_need_aware_prompt, validate_llm_response
from config.config_loader import get_config

# Import OpenAI integration from existing system
import sys
sys.path.append('../')
from utils import openai_api_key
import openai
openai.api_key = openai_api_key


class PersonaManager:
    """
    Manages predictive personas and their interactions with the environment
    """
    
    def __init__(self, use_predictive: bool = True):
        """
        Initialize the persona manager
        
        Args:
            use_predictive: Whether to use predictive personas or original
        """
        self.use_predictive = use_predictive
        self.agents: Dict[str, Any] = {}  # agent_name -> agent instance
        self.config = get_config()
        
        # Event queue for processing
        self.event_queue = asyncio.Queue()
        
        # Monitoring data
        self.monitoring_data = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        
    def create_agent(self, name: str, folder_mem_saved: Optional[str] = None) -> Any:
        """
        Create a new agent (predictive or original)
        
        Args:
            name: Agent name
            folder_mem_saved: Path to saved memory folder
            
        Returns:
            Agent instance
        """
        if self.use_predictive:
            agent = PredictivePersona(name, folder_mem_saved)
        else:
            agent = Persona(name, folder_mem_saved)
        
        self.agents[name] = agent
        
        # Initialize monitoring data
        self.monitoring_data[name] = {
            'needs': {},
            'monologue': [],
            'predictions': [],
            'socialModels': []
        }
        
        return agent
    
    def get_agent(self, name: str) -> Optional[Any]:
        """Get agent by name"""
        return self.agents.get(name)
    
    async def update_agents(self):
        """
        Main update loop for all agents
        Runs think cycles and processes actions
        """
        tasks = []
        
        for name, agent in self.agents.items():
            if isinstance(agent, PredictivePersona):
                task = asyncio.create_task(self._update_predictive_agent(agent))
                tasks.append(task)
            else:
                # Handle original persona update
                task = asyncio.create_task(self._update_original_agent(agent))
                tasks.append(task)
        
        # Run all agent updates in parallel
        if tasks:
            await asyncio.gather(*tasks)
        
        # Send monitoring updates
        await self._send_monitoring_updates()
    
    async def _update_predictive_agent(self, agent: PredictivePersona):
        """Update a single predictive agent"""
        try:
            # Run think cycle
            action = agent.think_cycle()
            
            # Process any actions
            if action:
                await self._process_agent_action(agent, action)
            
            # Check for externalization
            external_thought = await self._check_externalization(agent)
            if external_thought:
                await self._process_external_speech(agent, external_thought)
            
            # Update monitoring data
            self._update_monitoring_data(agent)
            
        except Exception as e:
            print(f"Error updating agent {agent.name}: {e}")
    
    async def _update_original_agent(self, agent: Persona):
        """Update original persona (compatibility mode)"""
        # This would integrate with the existing persona update logic
        pass
    
    async def _process_agent_action(self, agent: PredictivePersona, action: Dict[str, Any]):
        """Process an action decided by an agent"""
        action_type = action.get('type')
        
        if action_type == 'initiate_conversation':
            # Find nearby agents
            nearby = self._find_nearby_agents(agent)
            if nearby:
                target = nearby[0]  # Pick first available
                await self._initiate_conversation(agent, target)
                
        elif action_type == 'ask_for_help':
            await self._broadcast_help_request(agent)
            
        elif action_type == 'offer_help':
            # Find agents who might need help
            needy_agents = self._find_agents_needing_help(agent)
            if needy_agents:
                await self._offer_help_to(agent, needy_agents[0])
                
        # Add more action handlers as needed
    
    async def _check_externalization(self, agent: PredictivePersona) -> Optional[str]:
        """
        Check if agent should externalize thoughts and generate speech
        """
        # Check conditions for externalization
        if (agent.needs.needs['connection'] < self.config.get_externalization_threshold() 
            and agent.nearby_agents):
            
            # Generate prompt for LLM
            prompt = create_need_aware_prompt(
                'externalize',
                agent_name=agent.name,
                internal_monologue=agent.monologue.get_recent_thoughts(10),
                need_states=agent.needs.needs,
                predictions=list(agent.prediction_buffer)[-5:],
                context={
                    'location': agent.current_location,
                    'nearby_agents': [a.name for a in agent.nearby_agents],
                    'recent_events': [e.get('type', 'unknown') for e in agent.recent_events][-5:]
                }
            )
            
            # Call LLM
            try:
                response = await self._call_llm(prompt)
                
                # Validate response
                is_valid, cleaned, issues = validate_llm_response(
                    response, 
                    agent.needs.needs,
                    {'nearby_agents': agent.nearby_agents}
                )
                
                if is_valid:
                    return cleaned
                else:
                    print(f"Invalid response for {agent.name}: {issues}")
                    
            except Exception as e:
                print(f"LLM error for {agent.name}: {e}")
        
        return None
    
    async def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with prompt"""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a realistic human-like agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "..."
    
    async def _process_external_speech(self, agent: PredictivePersona, speech: str):
        """Process externalized speech from an agent"""
        # Create speech event
        event = {
            'type': 'agent_speech',
            'speaker': agent.name,
            'content': speech,
            'location': agent.current_location,
            'timestamp': datetime.now()
        }
        
        # Notify nearby agents
        for other_agent in agent.nearby_agents:
            if isinstance(other_agent, PredictivePersona):
                other_agent.observe_event(event)
        
        # Add to event queue for environment processing
        await self.event_queue.put(event)
    
    def _find_nearby_agents(self, agent: PredictivePersona) -> List[PredictivePersona]:
        """Find agents near the given agent"""
        nearby = []
        for name, other in self.agents.items():
            if name != agent.name and isinstance(other, PredictivePersona):
                # Check if in same location or adjacent
                if self._are_agents_nearby(agent, other):
                    nearby.append(other)
        return nearby
    
    def _are_agents_nearby(self, agent1: PredictivePersona, agent2: PredictivePersona) -> bool:
        """Check if two agents are nearby"""
        # This would integrate with the spatial system
        # For now, simple check
        return agent1.current_location == agent2.current_location
    
    def _find_agents_needing_help(self, helper: PredictivePersona) -> List[PredictivePersona]:
        """Find agents who might need help based on their need states"""
        needy = []
        
        for name, agent in self.agents.items():
            if name != helper.name and isinstance(agent, PredictivePersona):
                # Check if any critical needs
                critical_needs = [
                    need for need, value in agent.needs.needs.items()
                    if value < self.config.get_critical_thresholds()[need]
                ]
                
                if critical_needs:
                    needy.append(agent)
        
        return needy
    
    async def _initiate_conversation(self, initiator: PredictivePersona, 
                                   target: PredictivePersona):
        """Handle conversation initiation between agents"""
        # Create interaction event
        event = {
            'type': 'conversation_starts',
            'initiator': initiator.name,
            'target': target.name,
            'location': initiator.current_location,
            'timestamp': datetime.now()
        }
        
        # Both agents observe the event
        initiator.observe_event(event)
        target.observe_event(event)
        
        # Generate responses using social interaction prompts
        for agent, other in [(initiator, target), (target, initiator)]:
            response = agent.interact_with_agent(other, 'greeting')
            
            if response.get('content'):
                await self._process_external_speech(agent, response['content'])
    
    async def _broadcast_help_request(self, agent: PredictivePersona):
        """Broadcast a help request from an agent"""
        event = {
            'type': 'help_request',
            'requester': agent.name,
            'location': agent.current_location,
            'timestamp': datetime.now()
        }
        
        # Notify all nearby agents
        for other in self._find_nearby_agents(agent):
            other.observe_event(event)
    
    async def _offer_help_to(self, helper: PredictivePersona, 
                           target: PredictivePersona):
        """Handle help offering between agents"""
        event = {
            'type': 'help_offered',
            'helper': helper.name,
            'target': target.name,
            'timestamp': datetime.now()
        }
        
        helper.observe_event(event)
        target.observe_event(event)
    
    def _update_monitoring_data(self, agent: PredictivePersona):
        """Update monitoring data for visualization"""
        self.monitoring_data[agent.name] = {
            'needs': agent.needs.needs.copy(),
            'monologue': [
                {'type': t['type'], 'content': t['content']}
                for t in list(agent.monologue.thoughts)[-10:]
            ],
            'predictions': [
                {
                    'event': pred.get('event', 'unknown'),
                    'probability': pred.get('probability', 0),
                    'impacts': self._format_impacts(pred.get('need_impacts', {}))
                }
                for pred in list(agent.prediction_buffer)[-5:]
            ],
            'socialModels': [
                f"{other_name}: {self._summarize_needs(model['estimated_needs'])}"
                for other_name, model in list(agent.other_agent_models.items())[:5]
            ]
        }
    
    def _format_impacts(self, impacts: Dict[str, float]) -> str:
        """Format need impacts for display"""
        parts = []
        for need, change in sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            if abs(change) > 0.05:
                direction = "↑" if change > 0 else "↓"
                parts.append(f"{need}{direction}")
        return ", ".join(parts)
    
    def _summarize_needs(self, needs: Dict[str, float]) -> str:
        """Summarize need states for display"""
        # Find most deficient need
        deficient = min(needs.items(), key=lambda x: x[1])
        if deficient[1] < 0.3:
            return f"needs {deficient[0]} (critical)"
        elif deficient[1] < 0.5:
            return f"needs {deficient[0]} (low)"
        else:
            return "appears satisfied"
    
    async def _send_monitoring_updates(self):
        """Send monitoring updates to connected websockets"""
        if self.websocket_connections:
            update_data = {
                'timestamp': datetime.now().isoformat(),
                'agents': self.monitoring_data
            }
            
            # Send to all connected clients
            disconnected = set()
            for ws in self.websocket_connections:
                try:
                    await ws.send(json.dumps(update_data))
                except:
                    disconnected.add(ws)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected
    
    def add_websocket(self, ws):
        """Add a websocket connection for monitoring"""
        self.websocket_connections.add(ws)
    
    def remove_websocket(self, ws):
        """Remove a websocket connection"""
        self.websocket_connections.discard(ws)
    
    async def save_all_agents(self, base_path: str):
        """Save all agent states"""
        for name, agent in self.agents.items():
            if isinstance(agent, PredictivePersona):
                filepath = f"{base_path}/{name}_state.json"
                agent.save_state(filepath)
    
    def get_agent_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get a summary of an agent's current state"""
        agent = self.agents.get(agent_name)
        if not agent or not isinstance(agent, PredictivePersona):
            return {}
        
        return {
            'name': agent.name,
            'needs': agent.needs.needs,
            'recent_thoughts': agent.monologue.get_recent_thoughts(5),
            'current_predictions': [
                {
                    'event': p.get('event'),
                    'probability': p.get('probability')
                }
                for p in list(agent.prediction_buffer)[-3:]
            ],
            'location': agent.current_location,
            'nearby_agents': [a.name for a in agent.nearby_agents]
        }


# Global manager instance
_manager_instance = None

def get_manager(use_predictive: bool = True) -> PersonaManager:
    """Get global persona manager instance"""
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = PersonaManager(use_predictive)
    
    return _manager_instance


async def run_simulation_step():
    """Run one step of the simulation"""
    manager = get_manager()
    await manager.update_agents()