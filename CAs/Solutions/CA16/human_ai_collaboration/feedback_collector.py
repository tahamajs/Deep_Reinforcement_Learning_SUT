"""
Human Feedback Collection and Integration

This module implements systems for collecting and integrating human feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from collections import deque
import time
from .preference_model import HumanFeedback, HumanPreference


@dataclass
class FeedbackSession:
    """Represents a session of human feedback."""
    session_id: str
    start_time: float
    end_time: float
    feedbacks: List[HumanFeedback]
    context: Dict[str, Any]


class HumanFeedbackCollector:
    """Collects and manages human feedback."""
    
    def __init__(self, max_feedback_history: int = 1000):
        self.max_feedback_history = max_feedback_history
        self.feedback_history = deque(maxlen=max_feedback_history)
        self.active_sessions = {}
        self.feedback_statistics = {
            'total_feedbacks': 0,
            'positive_feedbacks': 0,
            'negative_feedbacks': 0,
            'neutral_feedbacks': 0,
            'avg_confidence': 0.0
        }
    
    def start_session(self, session_id: str, context: Dict[str, Any] = None) -> str:
        """Start a new feedback session."""
        if context is None:
            context = {}
        
        session = FeedbackSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=0.0,
            feedbacks=[],
            context=context
        )
        
        self.active_sessions[session_id] = session
        return session_id
    
    def end_session(self, session_id: str) -> FeedbackSession:
        """End a feedback session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.end_time = time.time()
        
        # Add feedbacks to history
        for feedback in session.feedbacks:
            self.feedback_history.append(feedback)
            self._update_statistics(feedback)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        return session
    
    def add_feedback(self, session_id: str, feedback: HumanFeedback):
        """Add feedback to a session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.feedbacks.append(feedback)
    
    def get_feedback_history(self, limit: int = None) -> List[HumanFeedback]:
        """Get feedback history."""
        if limit is None:
            return list(self.feedback_history)
        return list(self.feedback_history)[-limit:]
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return self.feedback_statistics.copy()
    
    def _update_statistics(self, feedback: HumanFeedback):
        """Update feedback statistics."""
        self.feedback_statistics['total_feedbacks'] += 1
        
        if feedback.feedback_type == 'positive':
            self.feedback_statistics['positive_feedbacks'] += 1
        elif feedback.feedback_type == 'negative':
            self.feedback_statistics['negative_feedbacks'] += 1
        else:
            self.feedback_statistics['neutral_feedbacks'] += 1
        
        # Update average confidence
        total = self.feedback_statistics['total_feedbacks']
        current_avg = self.feedback_statistics['avg_confidence']
        new_avg = (current_avg * (total - 1) + feedback.feedback_value) / total
        self.feedback_statistics['avg_confidence'] = new_avg