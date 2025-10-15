"""Theory of Mind module for understanding user intentions"""

from typing import Any

from jeffrey.core.cognitive.base_module import BaseCognitiveModule


class TheoryOfMind(BaseCognitiveModule):
    """Understands user intentions and mental states"""

    def __init__(self, memory):
        super().__init__("TheoryOfMind")
        self.memory = memory
        self.user_models = {}

        # Intention patterns
        self.intention_keywords = {
            "question": ["?", "what", "why", "how", "when", "where", "who", "which"],
            "request": ["please", "could", "would", "can you", "will you", "help"],
            "statement": [".", "i think", "i believe", "it is"],
            "exclamation": ["!", "wow", "amazing", "terrible"],
            "gratitude": ["thank", "thanks", "appreciate"],
            "greeting": ["hello", "hi", "hey", "good morning", "goodbye"],
        }

    async def on_initialize(self):
        """Load user models from memory"""
        if self.memory:
            try:
                users = await self.memory.retrieve("user_model", limit=100)
                for user_data in users:
                    if isinstance(user_data, dict):
                        user_id = user_data.get("user_id")
                        if user_id:
                            self.user_models[user_id] = user_data.get("model", {})
                self.logger.info(f"Loaded {len(self.user_models)} user models")
            except:
                pass

    def validate_input(self, data: dict[str, Any]) -> bool:
        """Need text and preferably user_id"""
        return "text" in data or "message" in data

    async def on_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze user intention and mental state"""
        text = (data.get("text") or data.get("message", "")).lower()
        user_id = data.get("user_id", "unknown")

        # Detect intention
        intention = self._detect_intention(text)

        # Detect emotion (simple sentiment)
        emotion = self._detect_emotion(text)

        # Update user model
        if user_id not in self.user_models:
            self.user_models[user_id] = {
                "interactions": 0,
                "intentions": {},
                "emotions": {},
                "topics": [],
            }

        model = self.user_models[user_id]
        model["interactions"] += 1
        model["intentions"][intention] = model["intentions"].get(intention, 0) + 1
        model["emotions"][emotion] = model["emotions"].get(emotion, 0) + 1

        # Extract topics (simple keyword extraction)
        topics = [w for w in text.split() if len(w) > 5]
        if topics:
            model["topics"].extend(topics[:3])
            model["topics"] = model["topics"][-20:]  # Keep last 20

        # Save to memory periodically
        if model["interactions"] % 10 == 0 and self.memory:
            await self.memory.store({"type": "user_model", "user_id": user_id, "model": model})

        return {
            "intention": intention,
            "emotion": emotion,
            "confidence": 0.7,  # Simple heuristic
            "user_profile": {
                "interactions": model["interactions"],
                "primary_intention": max(model["intentions"].items(), key=lambda x: x[1])[0]
                if model["intentions"]
                else "unknown",
                "primary_emotion": max(model["emotions"].items(), key=lambda x: x[1])[0]
                if model["emotions"]
                else "neutral",
            },
        }

    def _detect_intention(self, text: str) -> str:
        """Detect primary intention from text"""
        text_lower = text.lower()

        # Check each intention type
        scores = {}
        for intention, keywords in self.intention_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intention] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "statement"  # Default

    def _detect_emotion(self, text: str) -> str:
        """Simple emotion detection"""
        text_lower = text.lower()

        positive = ["good", "great", "happy", "love", "awesome", "excellent", "wonderful"]
        negative = ["bad", "sad", "angry", "hate", "terrible", "awful", "horrible"]

        pos_score = sum(1 for word in positive if word in text_lower)
        neg_score = sum(1 for word in negative if word in text_lower)

        if pos_score > neg_score:
            return "positive"
        elif neg_score > pos_score:
            return "negative"
        return "neutral"

    async def on_shutdown(self):
        """Save all user models"""
        if self.memory:
            for user_id, model in self.user_models.items():
                await self.memory.store({"type": "user_model", "user_id": user_id, "model": model})
