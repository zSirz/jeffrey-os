"""AutoLearner module for pattern learning"""

from collections import defaultdict
from typing import Any

from jeffrey.core.cognitive.base_module import BaseCognitiveModule


class AutoLearner(BaseCognitiveModule):
    """Learns patterns from interactions"""

    def __init__(self, memory):
        super().__init__("AutoLearner")
        self.memory = memory
        self.patterns = defaultdict(int)
        self.sequences = []
        self.max_sequences = 100

    async def on_initialize(self):
        """Custom initialization"""
        # Load previous patterns if available
        if self.memory:
            try:
                memories = await self.memory.retrieve("learned_pattern", limit=50)
                for mem in memories:
                    if isinstance(mem, dict) and "pattern" in mem:
                        self.patterns[mem["pattern"]] += mem.get("frequency", 1)
                self.logger.info(f"Loaded {len(self.patterns)} patterns from memory")
            except:
                pass

    def validate_input(self, data: dict[str, Any]) -> bool:
        """Validate that we have text to learn from"""
        return "text" in data or "message" in data

    async def on_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Learn patterns from text"""
        text = data.get("text") or data.get("message", "")
        if not text:
            return {"learned": False, "reason": "No text to learn from"}

        # Tokenize
        words = text.lower().split()

        # Update word frequencies
        for word in words:
            if len(word) > 2:  # Skip very short words
                self.patterns[word] += 1

        # Learn sequences (bigrams, trigrams)
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                self.patterns[bigram] += 1

        # Store significant patterns
        if len(words) > 5 and self.memory:
            pattern = " ".join(words[:3])
            if self.patterns[pattern] > 2:  # Only store repeated patterns
                await self.memory.store(
                    {
                        "type": "learned_pattern",
                        "pattern": pattern,
                        "frequency": self.patterns[pattern],
                        "context": text[:100],
                    }
                )

        # Keep sequence for context
        self.sequences.append(text)
        if len(self.sequences) > self.max_sequences:
            self.sequences.pop(0)

        return {
            "learned": True,
            "unique_patterns": len(self.patterns),
            "total_observations": sum(self.patterns.values()),
            "top_patterns": self._get_top_patterns(5),
        }

    def _get_top_patterns(self, n: int) -> list:
        """Get top N most frequent patterns"""
        sorted_patterns = sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)
        return [{"pattern": p, "count": c} for p, c in sorted_patterns[:n]]

    async def on_shutdown(self):
        """Save state before shutdown - FIXED iteration bug"""
        if self.memory and self.patterns:
            # Save top patterns
            for item in self._get_top_patterns(20):
                if item["count"] > 1:
                    await self.memory.store(
                        {
                            "type": "learned_pattern",
                            "pattern": item["pattern"],
                            "frequency": item["count"],
                        }
                    )
