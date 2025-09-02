"""
Transition Pattern Learning System for Coherent Text Generation
Learn and apply transition patterns from Centering Theory for fluent text generation
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
from lgram.models.centering_theory import TransitionType, CenteringState, EnhancedCenteringTheory

@dataclass
class TransitionPattern:
    """Represents a learned transition pattern"""
    sequence: List[TransitionType]  # Sequence of transitions
    frequency: int  # How often this pattern occurs
    coherence_score: float  # Average coherence score for texts with this pattern
    context_centers: List[str]  # Common centers in this pattern
    sentence_length_avg: float  # Average sentence length in this pattern
    
class TransitionPatternLearner:
    """Learn transition patterns from high-quality texts for coherent generation"""
    
    def __init__(self, centering_analyzer: EnhancedCenteringTheory):
        self.centering = centering_analyzer
        
        # Pattern storage
        self.patterns = {}  # pattern_id -> TransitionPattern
        self.pattern_sequences = defaultdict(list)  # sequence -> [pattern_ids]
        self.bigram_transitions = defaultdict(int)  # (t1, t2) -> count
        self.trigram_transitions = defaultdict(int)  # (t1, t2, t3) -> count
        
        # Quality scoring
        self.transition_quality_scores = {
            TransitionType.CONTINUE: 1.0,      # Best for coherence
            TransitionType.RETAIN: 0.8,        # Good for coherence  
            TransitionType.SMOOTH_SHIFT: 0.6,  # Acceptable shift
            TransitionType.ROUGH_SHIFT: 0.3    # Poor for coherence
        }
        
        # Pattern generation parameters
        self.min_pattern_length = 3
        self.max_pattern_length = 8
        self.min_frequency_threshold = 2
        
    def learn_from_text(self, text: str, quality_score: float = 1.0) -> Dict[str, Any]:
        """Learn transition patterns from a high-quality text"""
        
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < self.min_pattern_length:
            return {"error": "Text too short for pattern learning"}
        
        # Analyze centering for each sentence
        centering_states = []
        self.centering.discourse_history = []  # Reset history
        
        for sentence in sentences:
            if sentence:
                state = self.centering.analyze_utterance(sentence + ".")
                centering_states.append(state)
        
        # Extract transition sequence
        transitions = [state.transition for state in centering_states if state.transition]
        
        if len(transitions) < self.min_pattern_length:
            return {"error": "Not enough transitions for pattern learning"}
        
        # Learn patterns of different lengths
        patterns_learned = 0
        
        for length in range(self.min_pattern_length, min(len(transitions) + 1, self.max_pattern_length + 1)):
            for i in range(len(transitions) - length + 1):
                pattern_seq = tuple(transitions[i:i + length])
                
                # Calculate pattern quality
                pattern_quality = self._calculate_pattern_quality(pattern_seq, quality_score)
                
                # Extract context information
                context_states = centering_states[i:i + length]
                context_centers = self._extract_context_centers(context_states)
                avg_sentence_length = self._calculate_avg_sentence_length(sentences[i:i + length])
                
                # Store or update pattern
                pattern_id = f"pattern_{len(pattern_seq)}_{hash(pattern_seq)}"
                
                if pattern_id in self.patterns:
                    # Update existing pattern
                    pattern = self.patterns[pattern_id]
                    pattern.frequency += 1
                    # Update coherence score with weighted average
                    total_score = pattern.coherence_score * (pattern.frequency - 1) + pattern_quality
                    pattern.coherence_score = total_score / pattern.frequency
                else:
                    # Create new pattern
                    self.patterns[pattern_id] = TransitionPattern(
                        sequence=list(pattern_seq),
                        frequency=1,
                        coherence_score=pattern_quality,
                        context_centers=context_centers,
                        sentence_length_avg=avg_sentence_length
                    )
                
                self.pattern_sequences[pattern_seq].append(pattern_id)
                patterns_learned += 1
        
        # Update n-gram transition models
        self._update_ngram_models(transitions)
        
        return {
            "sentences_processed": len(sentences),
            "transitions_extracted": len(transitions),
            "patterns_learned": patterns_learned,
            "transition_sequence": [t.value for t in transitions]
        }
    
    def _calculate_pattern_quality(self, pattern_seq: Tuple[TransitionType, ...], base_quality: float) -> float:
        """Calculate quality score for a transition pattern"""
        
        # Base score from transition types
        transition_scores = [self.transition_quality_scores[t] for t in pattern_seq]
        avg_transition_score = sum(transition_scores) / len(transition_scores)
        
        # Bonus for good patterns (more CONTINUE and RETAIN)
        good_transitions = sum(1 for t in pattern_seq if t in [TransitionType.CONTINUE, TransitionType.RETAIN])
        coherence_bonus = good_transitions / len(pattern_seq) * 0.2
        
        # Penalty for too many ROUGH_SHIFT
        rough_shifts = sum(1 for t in pattern_seq if t == TransitionType.ROUGH_SHIFT)
        rough_penalty = (rough_shifts / len(pattern_seq)) * 0.3
        
        final_score = base_quality * avg_transition_score + coherence_bonus - rough_penalty
        return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
    
    def _extract_context_centers(self, states: List[CenteringState]) -> List[str]:
        """Extract common centers from a sequence of centering states"""
        all_centers = []
        for state in states:
            if state.forward_centers:
                all_centers.extend(state.forward_centers[:3])  # Top 3 centers
        
        # Return most common centers
        center_counts = Counter(all_centers)
        return [center for center, count in center_counts.most_common(5)]
    
    def _calculate_avg_sentence_length(self, sentences: List[str]) -> float:
        """Calculate average sentence length in words"""
        if not sentences:
            return 0.0
        
        lengths = [len(sentence.split()) for sentence in sentences]
        return sum(lengths) / len(lengths)
    
    def _update_ngram_models(self, transitions: List[TransitionType]) -> None:
        """Update bigram and trigram transition models"""
        
        # Update bigrams
        for i in range(len(transitions) - 1):
            bigram = (transitions[i], transitions[i + 1])
            self.bigram_transitions[bigram] += 1
        
        # Update trigrams
        for i in range(len(transitions) - 2):
            trigram = (transitions[i], transitions[i + 1], transitions[i + 2])
            self.trigram_transitions[trigram] += 1
    
    def generate_coherent_transition_sequence(self, target_length: int, 
                                           start_transition: Optional[TransitionType] = None) -> List[TransitionType]:
        """Generate a coherent transition sequence based on learned patterns"""
        
        if not self.patterns:
            # Fallback to default coherent pattern
            return self._generate_default_pattern(target_length)
        
        # Find best patterns to use
        suitable_patterns = self._find_suitable_patterns(target_length)
        
        if not suitable_patterns:
            return self._generate_from_ngrams(target_length, start_transition)
        
        # Select pattern based on quality and frequency
        selected_pattern = self._select_best_pattern(suitable_patterns)
        
        # Generate sequence using selected pattern
        sequence = []
        pattern_seq = selected_pattern.sequence
        
        # Start with the pattern or specified transition
        if start_transition:
            sequence.append(start_transition)
            target_length -= 1
        
        # Fill with pattern repetitions and variations
        while len(sequence) < target_length:
            remaining = target_length - len(sequence)
            
            if remaining >= len(pattern_seq):
                # Add full pattern
                sequence.extend(pattern_seq)
            else:
                # Add partial pattern or use n-gram continuation
                sequence.extend(self._complete_with_ngrams(sequence, remaining))
        
        return sequence[:target_length]
    
    def _find_suitable_patterns(self, target_length: int) -> List[TransitionPattern]:
        """Find patterns suitable for the target length"""
        suitable = []
        
        for pattern in self.patterns.values():
            # Pattern should be high quality and frequent enough
            if (pattern.frequency >= self.min_frequency_threshold and 
                pattern.coherence_score >= 0.6 and
                len(pattern.sequence) <= target_length):
                suitable.append(pattern)
        
        # Sort by quality and frequency
        suitable.sort(key=lambda p: (p.coherence_score, p.frequency), reverse=True)
        return suitable
    
    def _select_best_pattern(self, patterns: List[TransitionPattern]) -> TransitionPattern:
        """Select best pattern using weighted random selection"""
        if len(patterns) == 1:
            return patterns[0]
        
        # Calculate weights (quality * frequency)
        weights = [p.coherence_score * p.frequency for p in patterns]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return patterns[0]
        
        # Weighted random selection
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return patterns[i]
        
        return patterns[-1]  # Fallback
    
    def _generate_from_ngrams(self, target_length: int, 
                            start_transition: Optional[TransitionType] = None) -> List[TransitionType]:
        """Generate sequence using n-gram models"""
        
        if not self.bigram_transitions:
            return self._generate_default_pattern(target_length)
        
        sequence = []
        
        # Start transition
        if start_transition:
            current = start_transition
        else:
            # Choose most common starting transition
            first_transitions = defaultdict(int)
            for (t1, t2), count in self.bigram_transitions.items():
                first_transitions[t1] += count
            current = max(first_transitions.items(), key=lambda x: x[1])[0]
        
        sequence.append(current)
        
        # Generate using bigram model
        while len(sequence) < target_length:
            next_candidates = []
            
            # Try trigram first if we have enough context
            if len(sequence) >= 2:
                prev_prev = sequence[-2]
                prev = sequence[-1]
                for (t1, t2, t3), count in self.trigram_transitions.items():
                    if t1 == prev_prev and t2 == prev:
                        next_candidates.extend([t3] * count)
            
            # Fall back to bigram
            if not next_candidates:
                for (t1, t2), count in self.bigram_transitions.items():
                    if t1 == current:
                        next_candidates.extend([t2] * count)
            
            if next_candidates:
                import random
                current = random.choice(next_candidates)
                sequence.append(current)
            else:
                # Fallback to default continuation
                sequence.extend(self._generate_default_pattern(target_length - len(sequence)))
                break
        
        return sequence[:target_length]
    
    def _complete_with_ngrams(self, current_sequence: List[TransitionType], remaining: int) -> List[TransitionType]:
        """Complete sequence using n-gram models"""
        if not current_sequence or remaining <= 0:
            return []
        
        return self._generate_from_ngrams(remaining, current_sequence[-1])
    
    def _generate_default_pattern(self, length: int) -> List[TransitionType]:
        """Generate a default coherent pattern when no learned patterns exist"""
        
        # Default pattern emphasizes coherence: mostly CONTINUE with some RETAIN
        pattern = []
        
        for i in range(length):
            if i == 0:
                pattern.append(TransitionType.CONTINUE)
            elif i % 4 == 0:  # Occasional topic shift
                pattern.append(TransitionType.SMOOTH_SHIFT)
            elif i % 7 == 0:  # Occasional retain
                pattern.append(TransitionType.RETAIN)
            else:
                pattern.append(TransitionType.CONTINUE)
        
        return pattern
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        
        if not self.patterns:
            return {"total_patterns": 0}
        
        total_patterns = len(self.patterns)
        avg_coherence = sum(p.coherence_score for p in self.patterns.values()) / total_patterns
        
        # Transition distribution
        transition_counts = defaultdict(int)
        for pattern in self.patterns.values():
            for transition in pattern.sequence:
                transition_counts[transition] += pattern.frequency
        
        # Pattern length distribution
        length_counts = defaultdict(int)
        for pattern in self.patterns.values():
            length_counts[len(pattern.sequence)] += 1
        
        return {
            "total_patterns": total_patterns,
            "average_coherence_score": avg_coherence,
            "transition_distribution": {t.value: count for t, count in transition_counts.items()},
            "pattern_length_distribution": dict(length_counts),
            "bigram_count": len(self.bigram_transitions),
            "trigram_count": len(self.trigram_transitions)
        }
    
    def save_patterns(self, filepath: str) -> None:
        """Save learned patterns to file"""
        data = {
            'patterns': {pid: {
                'sequence': [t.value for t in p.sequence],
                'frequency': p.frequency,
                'coherence_score': p.coherence_score,
                'context_centers': p.context_centers,
                'sentence_length_avg': p.sentence_length_avg
            } for pid, p in self.patterns.items()},
            'bigram_transitions': {f"{t1.value},{t2.value}": count 
                                 for (t1, t2), count in self.bigram_transitions.items()},
            'trigram_transitions': {f"{t1.value},{t2.value},{t3.value}": count 
                                  for (t1, t2, t3), count in self.trigram_transitions.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_patterns(self, filepath: str) -> None:
        """Load patterns from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load patterns
        self.patterns = {}
        for pid, pdata in data.get('patterns', {}).items():
            sequence = [TransitionType(t) for t in pdata['sequence']]
            self.patterns[pid] = TransitionPattern(
                sequence=sequence,
                frequency=pdata['frequency'],
                coherence_score=pdata['coherence_score'],
                context_centers=pdata['context_centers'],
                sentence_length_avg=pdata['sentence_length_avg']
            )
        
        # Load n-gram models
        self.bigram_transitions = defaultdict(int)
        for key, count in data.get('bigram_transitions', {}).items():
            t1_str, t2_str = key.split(',')
            t1, t2 = TransitionType(t1_str), TransitionType(t2_str)
            self.bigram_transitions[(t1, t2)] = count
        
        self.trigram_transitions = defaultdict(int)  
        for key, count in data.get('trigram_transitions', {}).items():
            t1_str, t2_str, t3_str = key.split(',')
            t1, t2, t3 = TransitionType(t1_str), TransitionType(t2_str), TransitionType(t3_str)
            self.trigram_transitions[(t1, t2, t3)] = count
