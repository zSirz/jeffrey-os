"""Curiosity Engine for exploration and question generation"""

import random
from typing import Any

from jeffrey.core.cognitive.base_module import BaseCognitiveModule


class CuriosityEngine(BaseCognitiveModule):
    """Drives exploration through curiosity"""

    def __init__(self, memory):
        super().__init__("CuriosityEngine")
        self.memory = memory
        self.topics_explored = set()
        self.pending_questions = []
        self.max_questions = 50

        # Question templates
        self.question_templates = [
            "What is the relationship between {} and {}?",
            "How does {} work?",
            "Why is {} important?",
            "What are the alternatives to {}?",
            "Can you explain more about {}?",
            "What happens when {}?",
            "What is the difference between {} and {}?",
        ]

    async def on_initialize(self):
        """Load exploration history"""
        if self.memory:
            try:
                explored = await self.memory.retrieve("explored_topic", limit=100)
                for item in explored:
                    if isinstance(item, dict) and "topic" in item:
                        self.topics_explored.add(item["topic"])
                self.logger.info(f"Loaded {len(self.topics_explored)} explored topics")
            except:
                pass

    async def on_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Generate curiosity-driven questions"""
        text = data.get("text") or data.get("message", "")

        # Extract potential topics (words > 4 chars that aren't common)
        words = text.split()
        potential_topics = [
            w.lower().strip(".,!?") for w in words if len(w) > 4 and w.lower() not in self.topics_explored
        ]

        # Select novel topics
        novel_topics = [t for t in potential_topics if t not in self.topics_explored][:3]

        questions_generated = []

        if novel_topics:
            # Mark as explored
            for topic in novel_topics:
                self.topics_explored.add(topic)

                # Generate questions
                if len(self.pending_questions) < self.max_questions:
                    template = random.choice(self.question_templates)

                    if "{}" in template and template.count("{}") == 1:
                        question = template.format(topic)
                    elif template.count("{}") == 2 and len(novel_topics) > 1:
                        other_topic = random.choice([t for t in novel_topics if t != topic])
                        question = template.format(topic, other_topic)
                    else:
                        question = f"Tell me more about {topic}"

                    self.pending_questions.append(question)
                    questions_generated.append(question)

            # Save to memory
            if self.memory:
                for topic in novel_topics[:2]:  # Save top 2
                    await self.memory.store({"type": "explored_topic", "topic": topic, "context": text[:200]})

        # Prune old questions
        if len(self.pending_questions) > self.max_questions:
            self.pending_questions = self.pending_questions[-self.max_questions :]

        # Calculate curiosity level
        curiosity_level = len(novel_topics) / max(len(potential_topics), 1) if potential_topics else 0

        return {
            "curious": len(novel_topics) > 0,
            "curiosity_level": curiosity_level,
            "novel_topics": novel_topics,
            "questions": questions_generated[:2],  # Return max 2 questions
            "total_explored": len(self.topics_explored),
            "pending_questions": len(self.pending_questions),
        }

    async def on_shutdown(self):
        """Save exploration state"""
        if self.memory and len(self.topics_explored) > 0:
            # Save recently explored topics
            recent = list(self.topics_explored)[-10:]
            for topic in recent:
                await self.memory.store({"type": "explored_topic", "topic": topic})
