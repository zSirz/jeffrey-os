#!/usr/bin/env python3
"""Create all remaining learning modules in batch"""

from pathlib import Path

# Base directory
base_dir = Path("src/jeffrey/core/learning")
base_dir.mkdir(parents=True, exist_ok=True)

# Module templates
modules = {
    "unified_curiosity_engine.py": '''"""Unified curiosity engine with exploration strategies"""
from typing import Dict, Any, List, Optional
import time
import random
from collections import deque

from jeffrey.utils.logger import get_logger, log_method

logger = get_logger("UnifiedCuriosityEngine")


class UnifiedCuriosityEngine:
    """Advanced curiosity engine with multi-strategy exploration"""

    def __init__(self):
        self.exploration_history = deque(maxlen=1000)
        self.knowledge_gaps = {}
        self.curiosity_level = 0.5
        self.exploration_strategies = {
            "depth_first": self._explore_depth_first,
            "breadth_first": self._explore_breadth_first,
            "random_walk": self._explore_random_walk,
            "uncertainty_driven": self._explore_uncertainty_driven
        }
        self.current_strategy = "uncertainty_driven"

        self.stats = {
            "questions_generated": 0,
            "concepts_explored": 0,
            "knowledge_gaps_found": 0,
            "exploration_depth": 0
        }

    async def initialize(self):
        """Initialize curiosity engine"""
        logger.info("🔍 UnifiedCuriosityEngine initialized")
        return self

    async def explore(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Explore concept with adaptive strategy"""
        concept = context.get("concept", "unknown")
        depth = context.get("depth", 1.0)

        # Select exploration strategy based on context
        if depth > 0.7:
            strategy = self.exploration_strategies["depth_first"]
        elif depth < 0.3:
            strategy = self.exploration_strategies["breadth_first"]
        else:
            strategy = self.exploration_strategies[self.current_strategy]

        # Generate exploration questions
        questions = await strategy(concept, depth)

        # Update statistics
        self.stats["questions_generated"] += len(questions)
        self.stats["concepts_explored"] += 1
        self.stats["exploration_depth"] = max(self.stats["exploration_depth"], depth)

        # Record exploration
        self.exploration_history.append({
            "concept": concept,
            "questions": questions,
            "strategy": self.current_strategy,
            "timestamp": time.time()
        })

        return {
            "questions": questions,
            "gaps": list(self.knowledge_gaps.keys())[:5],
            "curiosity_level": self.curiosity_level,
            "strategy_used": self.current_strategy
        }

    async def _explore_depth_first(self, concept: str, depth: float) -> List[str]:
        """Deep exploration of single concept"""
        questions = [
            f"What are the fundamental principles of {concept}?",
            f"How does {concept} work at the deepest level?",
            f"What are the underlying mechanisms of {concept}?",
            f"What causes {concept} to behave this way?",
            f"What are the theoretical foundations of {concept}?"
        ]
        return questions[:int(3 + depth * 2)]

    async def _explore_breadth_first(self, concept: str, depth: float) -> List[str]:
        """Broad exploration of related concepts"""
        questions = [
            f"What concepts are related to {concept}?",
            f"How does {concept} connect to other ideas?",
            f"What are similar concepts to {concept}?",
            f"What categories does {concept} belong to?",
            f"What are practical applications of {concept}?"
        ]
        return questions[:int(3 + depth * 2)]

    async def _explore_random_walk(self, concept: str, depth: float) -> List[str]:
        """Random exploratory questions"""
        all_questions = [
            f"What if {concept} worked differently?",
            f"How would you explain {concept} to a child?",
            f"What are common misconceptions about {concept}?",
            f"What's the history of {concept}?",
            f"Who discovered or invented {concept}?",
            f"What problems does {concept} solve?",
            f"What are the limitations of {concept}?"
        ]
        random.shuffle(all_questions)
        return all_questions[:int(3 + depth * 2)]

    async def _explore_uncertainty_driven(self, concept: str, depth: float) -> List[str]:
        """Explore based on uncertainty and knowledge gaps"""
        # Identify gaps
        if concept not in self.knowledge_gaps:
            self.knowledge_gaps[concept] = 1.0

        questions = []

        # High uncertainty questions
        if self.knowledge_gaps[concept] > 0.7:
            questions.extend([
                f"What exactly is {concept}?",
                f"Can you define {concept}?",
                f"What are the key features of {concept}?"
            ])

        # Medium uncertainty questions
        elif self.knowledge_gaps[concept] > 0.3:
            questions.extend([
                f"How is {concept} used in practice?",
                f"What are variations of {concept}?",
                f"When is {concept} most applicable?"
            ])

        # Low uncertainty questions (advanced)
        else:
            questions.extend([
                f"What are edge cases for {concept}?",
                f"How can {concept} be optimized?",
                f"What are future developments for {concept}?"
            ])

        self.stats["knowledge_gaps_found"] = len(self.knowledge_gaps)
        return questions[:int(3 + depth * 2)]

    async def update_knowledge(self, concept: str, confidence: float):
        """Update knowledge gaps based on learning"""
        if concept in self.knowledge_gaps:
            # Reduce gap based on confidence
            self.knowledge_gaps[concept] *= (1 - confidence * 0.5)

            # Remove if sufficiently understood
            if self.knowledge_gaps[concept] < 0.1:
                del self.knowledge_gaps[concept]

        # Adjust curiosity level
        self.curiosity_level = 0.3 + 0.7 * (len(self.knowledge_gaps) / (len(self.knowledge_gaps) + 10))

    async def get_exploration_suggestions(self) -> List[str]:
        """Get suggestions for next exploration"""
        # Sort gaps by uncertainty
        sorted_gaps = sorted(self.knowledge_gaps.items(), key=lambda x: x[1], reverse=True)

        suggestions = []
        for concept, gap in sorted_gaps[:5]:
            suggestions.append(f"Explore {concept} (gap: {gap:.2f})")

        return suggestions
''',
    "auto_learner.py": '''"""Auto-learner with adaptive response generation"""
from typing import Dict, Any, Optional
import random
import time

from jeffrey.utils.logger import get_logger, log_method
from jeffrey.core.memory.unified_memory import MemoryValidator

logger = get_logger("AutoLearner")


class AutoLearner:
    """Adaptive auto-learning with response generation"""

    def __init__(self):
        self.learning_strategies = {
            "analytical": self._generate_analytical,
            "creative": self._generate_creative,
            "practical": self._generate_practical,
            "theoretical": self._generate_theoretical
        }

        self.response_templates = {
            "question": [
                "Based on the patterns I've observed, {} appears to be {}. {}",
                "From my analysis, {} can be understood as {}. {}",
                "Looking at the context, {} suggests {}. {}"
            ],
            "action": [
                "To {} effectively, I recommend {}. {}",
                "The best approach for {} would be {}. {}",
                "Based on similar patterns, {} can be achieved by {}. {}"
            ],
            "explanation": [
                "{} works by {}. The key aspect is {}.",
                "The concept of {} involves {}. Essentially, {}.",
                "{} can be explained as {}. In practice, {}."
            ]
        }

        self.stats = {
            "responses_generated": 0,
            "strategies_used": {},
            "average_confidence": 0.5
        }

    async def initialize(self):
        """Initialize auto-learner"""
        logger.info("🤖 AutoLearner initialized")
        return self

    async def generate(self, context: Dict[str, Any]) -> str:
        """Generate adaptive response based on context"""
        intention = context.get("intention", {})
        patterns = context.get("patterns", [])
        exploration = context.get("exploration", {})
        context_info = context.get("context", {})

        # Select strategy based on intention type
        intention_type = intention.get("type", "unknown")

        if intention_type in ["question", "inquiry"]:
            strategy = "analytical"
        elif intention_type in ["creative", "ideation"]:
            strategy = "creative"
        elif intention_type in ["action", "task"]:
            strategy = "practical"
        else:
            strategy = "theoretical"

        # Generate response with selected strategy
        generator = self.learning_strategies[strategy]
        response = await generator(intention, patterns, exploration, context_info)

        # Sanitize response
        response = MemoryValidator.sanitize_text(response)

        # Update statistics
        self.stats["responses_generated"] += 1
        if strategy not in self.stats["strategies_used"]:
            self.stats["strategies_used"][strategy] = 0
        self.stats["strategies_used"][strategy] += 1

        logger.debug(f"Generated {strategy} response: {response[:100]}...")

        return response

    async def _generate_analytical(self, intention: dict, patterns: list,
                                  exploration: dict, context: dict) -> str:
        """Generate analytical response"""
        main_concept = intention.get("main_concept", "the topic")
        confidence = intention.get("confidence", 0.5)

        # Analyze patterns
        if patterns:
            top_pattern = patterns[0]
            pattern_insight = f"I notice a pattern related to {top_pattern.get('value', 'this concept')}"
        else:
            pattern_insight = "I'm analyzing this from first principles"

        # Build analytical response
        if intention.get("type") == "question":
            template = self.response_templates["question"][0]
            analysis = f"This involves {len(patterns)} identified patterns"
            response = template.format(main_concept, pattern_insight, analysis)
        else:
            response = f"Analyzing {main_concept}: {pattern_insight}. "
            if exploration.get("questions"):
                response += f"Key aspects to consider: {exploration['questions'][0]}"

        return response

    async def _generate_creative(self, intention: dict, patterns: list,
                                exploration: dict, context: dict) -> str:
        """Generate creative response"""
        main_concept = intention.get("main_concept", "this idea")

        creative_approaches = [
            f"What if we approached {main_concept} from a completely different angle?",
            f"Imagine {main_concept} as a living system that evolves...",
            f"Let's think of {main_concept} as a puzzle where each piece represents...",
            f"Consider {main_concept} through the lens of emergence and complexity..."
        ]

        response = random.choice(creative_approaches)

        if patterns:
            response += f" I see connections to {patterns[0].get('value', 'related concepts')}."

        if exploration.get("questions"):
            response += f" This raises an interesting question: {exploration['questions'][0]}"

        return response

    async def _generate_practical(self, intention: dict, patterns: list,
                                 exploration: dict, context: dict) -> str:
        """Generate practical response"""
        main_concept = intention.get("main_concept", "this task")
        action = intention.get("action", "proceed")

        if intention.get("type") == "action":
            template = self.response_templates["action"][0]
            approach = "following a systematic approach"
            details = f"This typically takes {context.get('complexity', 1):.0f} steps"
            response = template.format(action, approach, details)
        else:
            response = f"For practical implementation of {main_concept}: "

            steps = [
                "First, establish the foundation",
                "Next, implement the core functionality",
                "Then, test and refine",
                "Finally, optimize and document"
            ]

            num_steps = min(3, int(context.get("complexity", 1) * 2))
            response += ". ".join(steps[:num_steps]) + "."

        return response

    async def _generate_theoretical(self, intention: dict, patterns: list,
                                   exploration: dict, context: dict) -> str:
        """Generate theoretical response"""
        main_concept = intention.get("main_concept", "this concept")

        template = self.response_templates["explanation"][0]

        # Build theoretical explanation
        mechanism = "a combination of interconnected principles"
        key_aspect = "the relationship between its components"

        if patterns and len(patterns) > 2:
            mechanism = f"interaction between {patterns[0].get('value', 'elements')} and {patterns[1].get('value', 'factors')}"

        response = template.format(main_concept, mechanism, key_aspect)

        if context.get("domain") == "technical":
            response += " From a technical perspective, this involves systematic processing and optimization."

        return response

    async def learn_from_feedback(self, response: str, feedback: dict):
        """Learn from feedback on generated response"""
        success = feedback.get("success", False)
        strategy = feedback.get("strategy", "unknown")

        # Update strategy confidence
        if success:
            self.stats["average_confidence"] = 0.9 * self.stats["average_confidence"] + 0.1
        else:
            self.stats["average_confidence"] = 0.9 * self.stats["average_confidence"]

        logger.debug(f"Learned from feedback: success={success}, confidence={self.stats['average_confidence']:.2f}")
''',
    "contextual_learning_engine.py": '''"""Contextual learning engine with domain recognition"""
from typing import Dict, Any, List
import re
import time

from jeffrey.utils.logger import get_logger, log_method
from jeffrey.core.memory.unified_memory import MemoryValidator

logger = get_logger("ContextualLearningEngine")


class ContextualLearningEngine:
    """Advanced contextual analysis and learning"""

    def __init__(self):
        self.domain_patterns = {
            "technical": ["code", "function", "api", "database", "algorithm", "system", "debug", "error"],
            "scientific": ["hypothesis", "experiment", "data", "analysis", "theory", "research", "study"],
            "creative": ["design", "art", "music", "story", "create", "imagine", "aesthetic"],
            "philosophical": ["meaning", "existence", "consciousness", "reality", "truth", "ethics"],
            "practical": ["how to", "steps", "guide", "tutorial", "implement", "build", "fix"],
            "emotional": ["feel", "emotion", "happy", "sad", "worry", "excited", "love", "fear"]
        }

        self.complexity_indicators = {
            "simple": ["what is", "define", "explain", "describe"],
            "moderate": ["how does", "compare", "analyze", "evaluate"],
            "complex": ["design", "architect", "optimize", "theoretical", "advanced"]
        }

        self.context_history = []
        self.domain_confidence = {}

        self.stats = {
            "contexts_analyzed": 0,
            "domains_identified": {},
            "avg_complexity": 0.5
        }

    async def initialize(self):
        """Initialize contextual engine"""
        logger.info("🎯 ContextualLearningEngine initialized")
        return self

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text context and extract metadata"""
        # Sanitize input
        text = MemoryValidator.sanitize_text(text)
        text_lower = text.lower()

        # Identify domain
        domain_scores = {}
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score

        # Select primary domain
        if domain_scores:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            domain_confidence = min(0.9, domain_scores[primary_domain] / 5)
        else:
            primary_domain = "general"
            domain_confidence = 0.3

        # Assess complexity
        complexity = await self._assess_complexity(text_lower)

        # Extract key entities
        entities = self._extract_entities(text)

        # Determine type
        context_type = await self._determine_type(text_lower)

        # Build context
        context = {
            "type": context_type,
            "domain": primary_domain,
            "complexity": complexity,
            "entities": entities,
            "domain_confidence": domain_confidence,
            "timestamp": time.time()
        }

        # Update history
        self.context_history.append(context)
        if len(self.context_history) > 100:
            self.context_history.pop(0)

        # Update statistics
        self.stats["contexts_analyzed"] += 1
        if primary_domain not in self.stats["domains_identified"]:
            self.stats["domains_identified"][primary_domain] = 0
        self.stats["domains_identified"][primary_domain] += 1

        # Update average complexity
        alpha = 0.1
        self.stats["avg_complexity"] = (1 - alpha) * self.stats["avg_complexity"] + alpha * complexity

        logger.debug(f"Context analyzed: type={context_type}, domain={primary_domain}, complexity={complexity:.2f}")

        return context

    async def _assess_complexity(self, text: str) -> float:
        """Assess text complexity (0-1 scale)"""
        complexity = 0.5  # Base complexity

        # Check complexity indicators
        for level, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    if level == "simple":
                        complexity = 0.3
                    elif level == "moderate":
                        complexity = 0.6
                    elif level == "complex":
                        complexity = 0.9
                    break

        # Adjust based on length
        word_count = len(text.split())
        if word_count > 100:
            complexity += 0.1
        if word_count > 200:
            complexity += 0.1

        # Adjust based on technical terms
        technical_terms = ["algorithm", "architecture", "framework", "protocol", "optimization"]
        tech_count = sum(1 for term in technical_terms if term in text)
        complexity += tech_count * 0.05

        return min(1.0, complexity)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        entities = []

        # Extract capitalized words (potential entities)
        caps = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        entities.extend(caps[:5])

        # Extract quoted strings
        quotes = re.findall(r'"([^"]+)"', text)
        entities.extend(quotes[:3])

        # Extract code-like terms
        code_terms = re.findall(r'\b[a-z]+_[a-z]+\b|\b[a-z]+[A-Z][a-zA-Z]+\b', text)
        entities.extend(code_terms[:3])

        return entities[:10]  # Limit to 10 entities

    async def _determine_type(self, text: str) -> str:
        """Determine the type of context"""
        # Question detection
        if any(text.startswith(q) for q in ["what", "how", "why", "when", "where", "who", "which"]):
            return "question"

        # Command/action detection
        if any(word in text for word in ["create", "make", "build", "generate", "write", "implement"]):
            return "action"

        # Analysis request
        if any(word in text for word in ["analyze", "evaluate", "compare", "assess", "review"]):
            return "analysis"

        # Information statement
        if any(word in text for word in ["is", "are", "was", "were", "has", "have"]):
            return "statement"

        return "general"

    async def get_context_recommendations(self, current_context: dict) -> List[str]:
        """Get recommendations based on context"""
        recommendations = []

        domain = current_context.get("domain", "general")
        complexity = current_context.get("complexity", 0.5)

        # Domain-specific recommendations
        if domain == "technical":
            recommendations.extend([
                "Consider implementation details",
                "Think about edge cases",
                "Evaluate performance implications"
            ])
        elif domain == "creative":
            recommendations.extend([
                "Explore alternative approaches",
                "Consider aesthetic aspects",
                "Think about user experience"
            ])
        elif domain == "scientific":
            recommendations.extend([
                "Review relevant research",
                "Consider empirical evidence",
                "Evaluate methodology"
            ])

        # Complexity-based recommendations
        if complexity > 0.7:
            recommendations.append("Break down into smaller components")
        elif complexity < 0.3:
            recommendations.append("Consider expanding the scope")

        return recommendations[:5]

    async def adapt_learning_strategy(self, context: dict) -> str:
        """Adapt learning strategy based on context"""
        domain = context.get("domain", "general")
        complexity = context.get("complexity", 0.5)
        context_type = context.get("type", "general")

        # Select strategy based on context
        if context_type == "question" and complexity < 0.5:
            return "direct_answer"
        elif context_type == "question" and complexity >= 0.5:
            return "exploratory_analysis"
        elif context_type == "action":
            return "step_by_step"
        elif domain == "creative":
            return "divergent_thinking"
        elif domain == "technical":
            return "systematic_approach"
        else:
            return "adaptive_learning"
''',
}

# Create all module files
for filename, content in modules.items():
    filepath = base_dir / filename

    # Skip if file already exists
    if filepath.exists():
        print(f"⚠️  Skipping {filename} - already exists")
        continue

    # Write the module
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Created {filename}")

print("\n🎉 All learning modules created successfully!")
