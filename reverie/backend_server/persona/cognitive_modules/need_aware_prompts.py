"""
Need-Aware Prompting Module

This module handles LLM integration for the predictive persona,
incorporating need states, predictions, and internal monologue
into prompts for generating natural language.

Author: AI Playground Team
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import datetime
import numpy as np


class NeedAwarePromptEngine:
    """
    Generates prompts that incorporate need states and predictions
    for more psychologically realistic agent behavior
    """
    
    def __init__(self):
        self.prompt_templates = self._load_templates()
        self.need_descriptions = self._load_need_descriptions()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for different interaction types"""
        return {
            'externalize_thought': """You are {agent_name}, an agent with internal needs and desires.

Your recent internal monologue:
{internal_monologue}

Your current need states (0=empty, 1=fulfilled):
{need_states}

Your most pressing needs: {priority_needs}

You've been predicting these upcoming events:
{predictions}

Current context:
- Location: {location}
- Nearby agents: {nearby_agents}
- Recent events: {recent_events}

Based on your internal state and predictions, generate a single line of dialogue that:
1. Reflects your current need priorities
2. Aims to create events that will fulfill your predicted needs
3. Sounds natural and conversational
4. Is consistent with your recent thinking

Remember: You're not explicitly talking about your needs, but your words should be motivated by them.

Your response:""",

            'social_interaction': """You are {agent_name} interacting with {other_agent}.

Your internal state:
- Recent thoughts: {recent_thoughts}
- Current needs: {need_states}
- What you think {other_agent} needs: {other_needs}

Interaction context:
- Type: {interaction_type}
- Your prediction: This interaction will {predicted_impact}
- Your goal: Fulfill your need for {primary_need} while considering {other_agent}'s needs

Recent interaction history with {other_agent}:
{interaction_history}

Generate a response that:
1. Addresses the {interaction_type} appropriately
2. Subtly works toward fulfilling your {primary_need}
3. Shows awareness of {other_agent}'s potential needs
4. Maintains conversational flow

Your response:""",

            'need_driven_planning': """You are {agent_name} planning your next actions.

Internal analysis:
{internal_monologue}

Need deficiencies (higher = more urgent):
{need_deficiencies}

Predicted outcomes for possible actions:
{action_predictions}

Current constraints:
- Location: {location}
- Time: {time}
- Energy level: {energy}
- Social context: {social_context}

Based on your predictions and need states, describe your next action plan.
Focus on actions that will lead to events fulfilling your most deficient needs.
Be specific about what you'll do and why (internally).

Your plan:""",

            'reflection_on_prediction': """You are {agent_name} reflecting on a prediction error.

What you predicted:
- Event: {predicted_event}
- Expected need changes: {predicted_impacts}

What actually happened:
- Event: {actual_event}
- Actual need changes: {actual_impacts}

Prediction error magnitude: {error_magnitude}

Your recent thoughts about this:
{relevant_thoughts}

Generate an internal reflection that:
1. Acknowledges what was different than expected
2. Updates your understanding of cause and effect
3. Adjusts future predictions
4. Maintains psychological realism

Your reflection:""",

            'empathy_modeling': """You are {agent_name} observing {other_agent}'s behavior.

Observed behavior: {observed_behavior}
Context: {context}

Your model of {other_agent}'s needs:
{estimated_needs}

Your recent thoughts about {other_agent}:
{social_thoughts}

Based on their behavior, generate an internal thought about:
1. What needs might be driving their actions
2. How accurate your model of them is
3. How you might interact with them to mutual benefit

Your insight:"""
        }
    
    def _load_need_descriptions(self) -> Dict[str, str]:
        """Load descriptions of what each need represents"""
        return {
            'connection': "desire for meaningful relationships and belonging",
            'safety': "need for physical and emotional security",
            'approval': "desire for recognition and validation from others",
            'empathy': "need to understand and be understood emotionally",
            'fun': "desire for joy, play, and entertainment",
            'attention': "need to be noticed and acknowledged",
            'achievement': "desire to accomplish goals and feel competent",
            'autonomy': "need for independence and self-direction",
            'purpose': "desire for meaning and contribution to something larger"
        }
    
    def format_need_states(self, needs: Dict[str, float]) -> str:
        """Format need states for prompt inclusion"""
        formatted = []
        for need, value in sorted(needs.items(), key=lambda x: x[1]):
            description = self.need_descriptions.get(need, need)
            if value < 0.3:
                status = "critically low"
            elif value < 0.5:
                status = "low"
            elif value < 0.7:
                status = "moderate"
            elif value < 0.9:
                status = "satisfied"
            else:
                status = "fully satisfied"
            
            formatted.append(f"- {need} ({description}): {value:.2f} [{status}]")
        
        return "\n".join(formatted)
    
    def format_predictions(self, predictions: List[Dict[str, Any]]) -> str:
        """Format predictions for prompt inclusion"""
        if not predictions:
            return "No specific predictions at the moment."
        
        formatted = []
        for pred in predictions[:5]:  # Limit to top 5
            event = pred.get('event', 'unknown')
            probability = pred.get('probability', 0)
            impacts = pred.get('need_impacts', {})
            
            impact_str = []
            for need, change in sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "↑" if change > 0 else "↓"
                impact_str.append(f"{need}{direction}")
            
            formatted.append(
                f"- {event} ({probability:.0%} likely) → {', '.join(impact_str)}"
            )
        
        return "\n".join(formatted)
    
    def format_need_deficiencies(self, needs: Dict[str, float]) -> str:
        """Format need deficiencies for planning"""
        deficiencies = [(need, 1 - value) for need, value in needs.items()]
        deficiencies.sort(key=lambda x: x[1], reverse=True)
        
        formatted = []
        for need, deficiency in deficiencies:
            if deficiency > 0.7:
                urgency = "CRITICAL"
            elif deficiency > 0.5:
                urgency = "High"
            elif deficiency > 0.3:
                urgency = "Moderate"
            else:
                urgency = "Low"
            
            formatted.append(f"- {need}: {deficiency:.2f} [{urgency}]")
        
        return "\n".join(formatted)
    
    def generate_externalization_prompt(self, 
                                      agent_name: str,
                                      internal_monologue: List[str],
                                      need_states: Dict[str, float],
                                      predictions: List[Dict[str, Any]],
                                      context: Dict[str, Any]) -> str:
        """Generate prompt for externalizing internal thoughts"""
        
        priority_needs = sorted(need_states.items(), key=lambda x: x[1])[:3]
        priority_need_names = [f"{need} ({value:.2f})" for need, value in priority_needs]
        
        return self.prompt_templates['externalize_thought'].format(
            agent_name=agent_name,
            internal_monologue="\n".join(f"- {thought}" for thought in internal_monologue[-10:]),
            need_states=self.format_need_states(need_states),
            priority_needs=", ".join(priority_need_names),
            predictions=self.format_predictions(predictions),
            location=context.get('location', 'unknown'),
            nearby_agents=", ".join(context.get('nearby_agents', ['no one'])),
            recent_events=", ".join(context.get('recent_events', ['nothing notable'])[-3:])
        )
    
    def generate_social_prompt(self,
                             agent_name: str,
                             other_agent: str,
                             interaction_type: str,
                             recent_thoughts: List[str],
                             need_states: Dict[str, float],
                             other_needs: Dict[str, float],
                             predicted_impact: Dict[str, float],
                             interaction_history: List[Dict[str, Any]]) -> str:
        """Generate prompt for social interaction"""
        
        # Find primary need to fulfill
        deficiencies = {need: 1 - value for need, value in need_states.items()}
        primary_need = max(deficiencies.items(), key=lambda x: x[1])[0]
        
        # Format predicted impact
        impact_parts = []
        for need, change in predicted_impact.items():
            if abs(change) > 0.05:
                direction = "increase" if change > 0 else "decrease"
                impact_parts.append(f"{direction} your {need}")
        
        impact_str = " and ".join(impact_parts) if impact_parts else "have minimal impact"
        
        # Format interaction history
        history_str = []
        for interaction in interaction_history[-3:]:
            history_str.append(
                f"- {interaction['type']}: {interaction.get('summary', 'brief exchange')}"
            )
        
        return self.prompt_templates['social_interaction'].format(
            agent_name=agent_name,
            other_agent=other_agent,
            recent_thoughts="\n".join(f"- {t}" for t in recent_thoughts[-5:]),
            need_states=self.format_need_states(need_states),
            other_needs=self.format_need_states(other_needs),
            interaction_type=interaction_type,
            predicted_impact=impact_str,
            primary_need=primary_need,
            interaction_history="\n".join(history_str) if history_str else "No recent interactions"
        )
    
    def generate_planning_prompt(self,
                               agent_name: str,
                               internal_monologue: List[str],
                               needs: Dict[str, float],
                               action_predictions: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> str:
        """Generate prompt for action planning"""
        
        # Format action predictions with expected outcomes
        formatted_predictions = []
        for pred in action_predictions[:5]:
            action = pred.get('action', 'unknown')
            expected_outcome = pred.get('expected_outcome', {})
            confidence = pred.get('confidence', 0.5)
            
            outcome_str = []
            for need, change in expected_outcome.items():
                if abs(change) > 0.05:
                    direction = "+" if change > 0 else ""
                    outcome_str.append(f"{need}: {direction}{change:.2f}")
            
            formatted_predictions.append(
                f"- {action} → {', '.join(outcome_str)} (confidence: {confidence:.0%})"
            )
        
        return self.prompt_templates['need_driven_planning'].format(
            agent_name=agent_name,
            internal_monologue="\n".join(internal_monologue[-10:]),
            need_deficiencies=self.format_need_deficiencies(needs),
            action_predictions="\n".join(formatted_predictions),
            location=context.get('location', 'unknown'),
            time=context.get('time', datetime.datetime.now().strftime("%H:%M")),
            energy=context.get('energy', 0.8),
            social_context=context.get('social_context', 'alone')
        )
    
    def generate_reflection_prompt(self,
                                 agent_name: str,
                                 predicted_event: str,
                                 actual_event: str,
                                 predicted_impacts: Dict[str, float],
                                 actual_impacts: Dict[str, float],
                                 relevant_thoughts: List[str]) -> str:
        """Generate prompt for reflecting on prediction errors"""
        
        # Calculate error magnitude
        error_magnitude = sum(
            abs(actual_impacts.get(need, 0) - predicted_impacts.get(need, 0))
            for need in set(list(actual_impacts.keys()) + list(predicted_impacts.keys()))
        )
        
        # Format impacts
        def format_impacts(impacts):
            return ", ".join(
                f"{need}: {'+' if value > 0 else ''}{value:.2f}"
                for need, value in sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            )
        
        return self.prompt_templates['reflection_on_prediction'].format(
            agent_name=agent_name,
            predicted_event=predicted_event,
            actual_event=actual_event,
            predicted_impacts=format_impacts(predicted_impacts),
            actual_impacts=format_impacts(actual_impacts),
            error_magnitude=f"{error_magnitude:.2f}",
            relevant_thoughts="\n".join(f"- {t}" for t in relevant_thoughts[-5:])
        )
    
    def generate_empathy_prompt(self,
                              agent_name: str,
                              other_agent: str,
                              observed_behavior: str,
                              context: Dict[str, Any],
                              estimated_needs: Dict[str, float],
                              social_thoughts: List[str]) -> str:
        """Generate prompt for empathetic understanding"""
        
        return self.prompt_templates['empathy_modeling'].format(
            agent_name=agent_name,
            other_agent=other_agent,
            observed_behavior=observed_behavior,
            context=json.dumps(context, indent=2),
            estimated_needs=self.format_need_states(estimated_needs),
            social_thoughts="\n".join(f"- {t}" for t in social_thoughts[-5:])
        )
    
    def extract_response(self, llm_output: str) -> str:
        """Extract clean response from LLM output"""
        # Remove any meta-commentary or system messages
        lines = llm_output.strip().split('\n')
        
        # Find the actual response (usually after "Your response:" or similar)
        response_start = -1
        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in ['response:', 'plan:', 'reflection:', 'insight:']):
                response_start = i + 1
                break
        
        if response_start > 0 and response_start < len(lines):
            return '\n'.join(lines[response_start:]).strip()
        
        # Fallback: return cleaned full output
        return llm_output.strip()


class ResponseValidator:
    """
    Validates and potentially modifies LLM responses to ensure
    they align with the agent's need states and predictions
    """
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load rules for validating responses"""
        return {
            'max_length': 150,  # Maximum response length in characters
            'min_length': 10,   # Minimum response length
            'forbidden_phrases': [
                'as an ai', 'as a language model', 'i am programmed',
                'my needs are', 'i predict that', 'my internal state'
            ],
            'need_consistency_threshold': 0.7,  # How well response should align with needs
        }
    
    def validate_response(self, 
                         response: str, 
                         need_states: Dict[str, float],
                         context: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        """
        Validate a response against need states and context
        Returns: (is_valid, cleaned_response, issues)
        """
        issues = []
        cleaned = response.strip()
        
        # Check length
        if len(cleaned) > self.validation_rules['max_length']:
            issues.append("Response too long")
            cleaned = cleaned[:self.validation_rules['max_length']] + "..."
        elif len(cleaned) < self.validation_rules['min_length']:
            issues.append("Response too short")
            return False, cleaned, issues
        
        # Check for forbidden phrases
        lower_response = cleaned.lower()
        for phrase in self.validation_rules['forbidden_phrases']:
            if phrase in lower_response:
                issues.append(f"Contains forbidden phrase: '{phrase}'")
                # Attempt to remove or rephrase
                cleaned = cleaned.replace(phrase, "")
        
        # Check need consistency
        if not self._check_need_consistency(cleaned, need_states, context):
            issues.append("Response inconsistent with need states")
        
        is_valid = len(issues) == 0
        return is_valid, cleaned, issues
    
    def _check_need_consistency(self, 
                               response: str, 
                               need_states: Dict[str, float],
                               context: Dict[str, Any]) -> bool:
        """
        Check if response is consistent with agent's need states
        This is a simplified version - full implementation would use
        sentiment analysis and more sophisticated NLP
        """
        # Get most deficient needs
        deficient_needs = sorted(
            [(need, 1 - value) for need, value in need_states.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Simple keyword checking (would be more sophisticated in practice)
        need_keywords = {
            'connection': ['together', 'join', 'chat', 'talk', 'friend'],
            'safety': ['careful', 'safe', 'worry', 'concern', 'protect'],
            'approval': ['like', 'appreciate', 'great', 'wonderful', 'impressed'],
            'empathy': ['understand', 'feel', 'sorry', 'relate', 'imagine'],
            'fun': ['enjoy', 'play', 'laugh', 'exciting', 'interesting'],
            'attention': ['notice', 'see', 'look', 'watch', 'observe'],
            'achievement': ['accomplish', 'succeed', 'complete', 'finish', 'goal'],
            'autonomy': ['decide', 'choose', 'myself', 'own', 'independent'],
            'purpose': ['help', 'contribute', 'meaningful', 'important', 'matter']
        }
        
        # Check if response addresses deficient needs
        response_lower = response.lower()
        addressed_needs = 0
        
        for need, deficiency in deficient_needs:
            if need in need_keywords:
                keywords = need_keywords[need]
                if any(keyword in response_lower for keyword in keywords):
                    addressed_needs += 1
        
        # Consider consistent if addressing at least one deficient need
        return addressed_needs > 0 or len(deficient_needs) == 0


def create_need_aware_prompt(prompt_type: str, **kwargs) -> str:
    """
    Factory function to create need-aware prompts
    """
    engine = NeedAwarePromptEngine()
    
    prompt_generators = {
        'externalize': engine.generate_externalization_prompt,
        'social': engine.generate_social_prompt,
        'planning': engine.generate_planning_prompt,
        'reflection': engine.generate_reflection_prompt,
        'empathy': engine.generate_empathy_prompt
    }
    
    if prompt_type not in prompt_generators:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompt_generators[prompt_type](**kwargs)


def validate_llm_response(response: str, 
                         need_states: Dict[str, float],
                         context: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    """
    Validate and clean LLM response
    """
    validator = ResponseValidator()
    return validator.validate_response(response, need_states, context)