"""
Django views for the Need-Based Agent Monitor

This module provides views and WebSocket endpoints for the agent monitoring
interface, allowing real-time observation of agent states and reasoning.

Author: AI Playground Team
"""

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from channels.generic.websocket import AsyncWebsocketConsumer
import json
import asyncio
from datetime import datetime

# Import persona manager from backend
import sys
import os
backend_path = os.path.join(os.path.dirname(__file__), '../../reverie/backend_server')
sys.path.append(backend_path)
from persona.persona_manager import get_manager


def agent_monitor_view(request):
    """
    Render the agent monitor interface
    """
    return render(request, 'agent_monitor.html')


@csrf_exempt
def get_agent_states(request):
    """
    API endpoint to get current agent states
    """
    if request.method == 'GET':
        try:
            manager = get_manager()
            
            # Collect agent data
            agent_data = {}
            for name in manager.monitoring_data:
                agent_data[name] = manager.monitoring_data[name]
            
            return JsonResponse({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'agents': agent_data
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def get_agent_details(request, agent_name):
    """
    Get detailed information about a specific agent
    """
    if request.method == 'GET':
        try:
            manager = get_manager()
            summary = manager.get_agent_summary(agent_name)
            
            if summary:
                return JsonResponse({
                    'status': 'success',
                    'agent': summary
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Agent {agent_name} not found'
                }, status=404)
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def update_agent_need(request, agent_name):
    """
    Manually update an agent's need (for educational demonstrations)
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            need_name = data.get('need')
            change_value = data.get('change', 0)
            
            manager = get_manager()
            agent = manager.get_agent(agent_name)
            
            if agent and hasattr(agent, 'needs'):
                old_value = agent.needs.needs.get(need_name, 0)
                agent.needs.update(need_name, change_value)
                new_value = agent.needs.needs.get(need_name, 0)
                
                # Log the manual intervention
                agent.monologue.add_thought(
                    'observation',
                    content=f"*External intervention: {need_name} changed by {change_value:.2f}*"
                )
                
                return JsonResponse({
                    'status': 'success',
                    'need': need_name,
                    'old_value': old_value,
                    'new_value': new_value,
                    'change': change_value
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Agent {agent_name} not found or not predictive'
                }, status=404)
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def trigger_agent_event(request):
    """
    Trigger an event for agents to observe (for educational demonstrations)
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            event_type = data.get('event_type')
            location = data.get('location', 'town_square')
            
            # Create event
            event = {
                'type': event_type,
                'location': location,
                'timestamp': datetime.now(),
                'source': 'manual_trigger'
            }
            
            # Add any additional event data
            for key, value in data.items():
                if key not in ['event_type', 'location']:
                    event[key] = value
            
            # Notify agents at location
            manager = get_manager()
            notified_agents = []
            
            for name, agent in manager.agents.items():
                if hasattr(agent, 'current_location') and agent.current_location == location:
                    agent.observe_event(event)
                    notified_agents.append(name)
            
            return JsonResponse({
                'status': 'success',
                'event': event,
                'notified_agents': notified_agents
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


class AgentMonitorConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time agent monitoring
    """
    
    async def connect(self):
        """Handle WebSocket connection"""
        self.room_group_name = 'agent_monitor'
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Register with persona manager
        manager = get_manager()
        manager.add_websocket(self)
        
        # Send initial state
        await self.send_agent_states()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        
        # Unregister from persona manager
        manager = get_manager()
        manager.remove_websocket(self)
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'request_update':
                await self.send_agent_states()
            elif message_type == 'pause_agent':
                # Implement agent pausing for detailed inspection
                agent_name = data.get('agent_name')
                # TODO: Implement pause functionality
            elif message_type == 'resume_agent':
                # Implement agent resuming
                agent_name = data.get('agent_name')
                # TODO: Implement resume functionality
                
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def send_agent_states(self):
        """Send current agent states to client"""
        try:
            manager = get_manager()
            
            agent_data = {}
            for name in manager.monitoring_data:
                agent_data[name] = manager.monitoring_data[name]
            
            await self.send(text_data=json.dumps({
                'type': 'agent_update',
                'timestamp': datetime.now().isoformat(),
                'agents': agent_data
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    # Receive message from room group
    async def agent_update(self, event):
        """Handle agent update broadcasts"""
        # Send message to WebSocket
        await self.send(text_data=json.dumps(event['data']))


# Educational interface endpoints

@csrf_exempt
def get_prediction_explanation(request, agent_name):
    """
    Get detailed explanation of an agent's current predictions
    """
    if request.method == 'GET':
        try:
            manager = get_manager()
            agent = manager.get_agent(agent_name)
            
            if agent and hasattr(agent, 'prediction_buffer'):
                # Get recent predictions with explanations
                predictions = []
                for pred in list(agent.prediction_buffer)[-5:]:
                    explanation = {
                        'event': pred.get('event'),
                        'probability': pred.get('probability'),
                        'reasoning': f"Based on recent events and current needs, "
                                   f"I estimate {pred.get('probability', 0)*100:.0f}% "
                                   f"chance of {pred.get('event')}",
                        'expected_impacts': pred.get('need_impacts', {}),
                        'confidence': pred.get('confidence', 0.5)
                    }
                    predictions.append(explanation)
                
                return JsonResponse({
                    'status': 'success',
                    'agent': agent_name,
                    'predictions': predictions
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Agent {agent_name} not found or not predictive'
                }, status=404)
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def get_learning_progress(request, agent_name):
    """
    Get information about an agent's learning progress
    """
    if request.method == 'GET':
        try:
            manager = get_manager()
            agent = manager.get_agent(agent_name)
            
            if agent and hasattr(agent, 'prediction_errors'):
                # Calculate learning metrics
                recent_errors = list(agent.prediction_errors)[-20:]
                
                if recent_errors:
                    avg_error = sum(e['error'] for e in recent_errors) / len(recent_errors)
                    error_trend = 'improving' if recent_errors[-1]['error'] < recent_errors[0]['error'] else 'stable'
                else:
                    avg_error = 0
                    error_trend = 'no data'
                
                # Get event vocabulary size (learning breadth)
                vocab_size = len(agent.prediction_model.event_vocabulary)
                
                return JsonResponse({
                    'status': 'success',
                    'agent': agent_name,
                    'learning_metrics': {
                        'average_prediction_error': avg_error,
                        'error_trend': error_trend,
                        'event_vocabulary_size': vocab_size,
                        'total_predictions': len(agent.prediction_buffer),
                        'total_errors_tracked': len(agent.prediction_errors)
                    }
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Agent {agent_name} not found or not predictive'
                }, status=404)
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)