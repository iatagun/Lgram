import spacy
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TransitionType(Enum):
    """Centering Theory Transition Types"""
    CONTINUE = "Continue"           # Cb(Ui) = Cb(Ui-1) = Cp(Ui)
    RETAIN = "Retain"              # Cb(Ui) = Cb(Ui-1) ≠ Cp(Ui)  
    SMOOTH_SHIFT = "Smooth-Shift"   # Cb(Ui) ≠ Cb(Ui-1) = Cp(Ui)
    ROUGH_SHIFT = "Rough-Shift"     # Cb(Ui) ≠ Cb(Ui-1) ≠ Cp(Ui)

@dataclass
class CenteringState:
    """Represents centering state for an utterance"""
    utterance: str
    forward_centers: List[str]  # Cf - ordered by salience
    backward_center: Optional[str]  # Cb
    preferred_center: Optional[str]  # Cp (= Cf[0])
    transition: Optional[TransitionType] = None

class EnhancedCenteringTheory:
    """Enhanced Centering Theory Implementation for Coherent Text Generation"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.discourse_history: List[CenteringState] = []
        
        # Salience weights for different grammatical roles
        self.salience_weights = {
            'nsubj': 4,      # Subject - highest salience
            'nsubjpass': 4,   # Passive subject
            'dobj': 3,        # Direct object
            'iobj': 2,        # Indirect object
            'pobj': 2,        # Object of preposition
            'attr': 2,        # Attribute
            'oprd': 2,        # Object predicate
            'poss': 1,        # Possessive
            'appos': 1        # Apposition
        }
        
        # POS-based salience for pronouns and proper nouns
        self.pos_weights = {
            'PRON': 3,        # Pronouns are highly salient
            'PROPN': 2,       # Proper nouns
            'NOUN': 1         # Common nouns
        }
    
    def compute_forward_centers(self, utterance: str) -> List[str]:
        """
        Compute Cf (Forward-looking Centers) ordered by salience
        Uses grammatical role, position, and entity type
        """
        doc = self.nlp(utterance)
        centers = []
        
        for token in doc:
            if self._is_center_candidate(token):
                salience = self._calculate_salience(token, utterance)
                centers.append((token.text.lower(), salience, token.i))
        
        # Sort by salience (desc), then by position (asc)
        centers.sort(key=lambda x: (-x[1], x[2]))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_centers = []
        for center, _, _ in centers:
            if center not in seen:
                seen.add(center)
                unique_centers.append(center)
        
        return unique_centers[:5]  # Limit to top 5 centers
    
    def _is_center_candidate(self, token) -> bool:
        """Determine if token can be a discourse center"""
        return (
            token.pos_ in ['NOUN', 'PROPN', 'PRON'] and
            not token.is_stop and
            not token.is_punct and
            len(token.text) > 1 and
            token.dep_ not in ['det', 'aux', 'auxpass', 'cop']
        )
    
    def _calculate_salience(self, token, utterance: str) -> float:
        """Calculate salience score for a token"""
        score = 0.0
        
        # Grammatical role weight
        if token.dep_ in self.salience_weights:
            score += self.salience_weights[token.dep_]
        
        # POS weight
        if token.pos_ in self.pos_weights:
            score += self.pos_weights[token.pos_]
        
        # Position weight (earlier = more salient)
        position_weight = 1.0 - (token.i / len(utterance.split()))
        score += position_weight
        
        # Entity type bonus
        if token.ent_type_ in ['PERSON', 'ORG', 'GPE']:
            score += 1.5
        
        # Pronoun resolution bonus
        if token.pos_ == 'PRON' and self._has_clear_antecedent(token):
            score += 1.0
        
        return score
    
    def _has_clear_antecedent(self, pronoun_token) -> bool:
        """Check if pronoun has clear antecedent in recent discourse"""
        if not self.discourse_history:
            return False
        
        # Look for matching entities in recent utterances
        recent_centers = []
        for state in self.discourse_history[-3:]:  # Last 3 utterances
            recent_centers.extend(state.forward_centers)
        
        return len(recent_centers) > 0
    
    def compute_backward_center(self, current_cf: List[str]) -> Optional[str]:
        """
        Compute Cb (Backward-looking Center)
        Cb(Ui) = highest ranked element of Cf(Ui-1) that is realized in Ui
        """
        if not self.discourse_history or not current_cf:
            return None
        
        previous_cf = self.discourse_history[-1].forward_centers
        
        # Find highest-ranked element from previous Cf that appears in current Cf
        for prev_center in previous_cf:
            if prev_center in current_cf:
                return prev_center
        
        return None
    
    def determine_transition(self, current_state: CenteringState) -> TransitionType:
        """Determine transition type based on centering theory rules"""
        if not self.discourse_history:
            return TransitionType.CONTINUE  # First utterance
        
        prev_state = self.discourse_history[-1]
        
        current_cb = current_state.backward_center
        prev_cb = prev_state.backward_center
        current_cp = current_state.preferred_center
        
        # Apply centering theory transition rules
        if current_cb == prev_cb:
            if current_cb == current_cp:
                return TransitionType.CONTINUE
            else:
                return TransitionType.RETAIN
        else:
            if prev_cb == current_cp:
                return TransitionType.SMOOTH_SHIFT
            else:
                return TransitionType.ROUGH_SHIFT
    
    def analyze_utterance(self, utterance: str) -> CenteringState:
        """Analyze utterance and compute centering information"""
        cf = self.compute_forward_centers(utterance)
        cb = self.compute_backward_center(cf)
        cp = cf[0] if cf else None
        
        state = CenteringState(
            utterance=utterance,
            forward_centers=cf,
            backward_center=cb,
            preferred_center=cp
        )
        
        state.transition = self.determine_transition(state)
        return state
    
    def update_discourse(self, utterance: str) -> CenteringState:
        """Update discourse history with new utterance"""
        state = self.analyze_utterance(utterance)
        self.discourse_history.append(state)
        
        # Keep only recent history to avoid memory issues
        if len(self.discourse_history) > 10:
            self.discourse_history = self.discourse_history[-10:]
        
        return state
    
    def get_coherent_next_center(self, preference_weight: float = 0.8) -> Optional[str]:
        """
        Get next coherent center based on centering preferences
        Prefers CONTINUE > RETAIN > SMOOTH-SHIFT > ROUGH-SHIFT
        """
        if not self.discourse_history:
            return None
        
        current_state = self.discourse_history[-1]
        
        # Prefer continuing current center
        if current_state.backward_center:
            return current_state.backward_center
        
        # Otherwise, prefer highest-ranked forward center
        if current_state.forward_centers:
            return current_state.forward_centers[0]
        
        return None
    
    def evaluate_coherence(self, utterance_sequence: List[str]) -> Dict[str, float]:
        """Evaluate discourse coherence based on centering patterns"""
        if len(utterance_sequence) < 2:
            return {"coherence_score": 1.0, "transition_distribution": {}}
        
        # Reset discourse history for evaluation
        self.discourse_history = []
        transition_counts = {t: 0 for t in TransitionType}
        
        for utterance in utterance_sequence:
            state = self.update_discourse(utterance)
            if state.transition:
                transition_counts[state.transition] += 1
        
        total_transitions = sum(transition_counts.values())
        if total_transitions == 0:
            return {"coherence_score": 1.0, "transition_distribution": {}}
        
        # Calculate coherence score based on transition preferences
        transition_weights = {
            TransitionType.CONTINUE: 1.0,
            TransitionType.RETAIN: 0.8,
            TransitionType.SMOOTH_SHIFT: 0.6,
            TransitionType.ROUGH_SHIFT: 0.3
        }
        
        weighted_score = sum(
            transition_counts[t] * transition_weights[t] 
            for t in TransitionType
        ) / total_transitions
        
        transition_distribution = {
            t.value: transition_counts[t] / total_transitions 
            for t in TransitionType
        }
        
        return {
            "coherence_score": weighted_score,
            "transition_distribution": transition_distribution,
            "total_transitions": total_transitions
        }
    
    def get_discourse_summary(self) -> Dict:
        """Get summary of current discourse state"""
        if not self.discourse_history:
            return {"message": "No discourse history"}
        
        recent_centers = []
        recent_transitions = []
        
        for state in self.discourse_history[-5:]:
            if state.forward_centers:
                recent_centers.extend(state.forward_centers[:2])
            if state.transition:
                recent_transitions.append(state.transition.value)
        
        return {
            "recent_centers": list(set(recent_centers)),
            "recent_transitions": recent_transitions,
            "current_cb": self.discourse_history[-1].backward_center,
            "current_cp": self.discourse_history[-1].preferred_center,
            "discourse_length": len(self.discourse_history)
        }