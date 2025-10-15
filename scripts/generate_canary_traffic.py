#!/usr/bin/env python3
"""
Generate realistic local canary traffic for Jeffrey OS.

Features:
- Weighted pools: 70% normal, 20% borderline, 5% PII, 5% edge
- PII rate limiting: max 2/min to avoid metric pollution
- Mixed FR/EN, registers, emojis, sarcasm
- Explicit anger/frustration tie-break triggers
- Low-confidence borderline cases
- Variable lengths & punctuation / noise
- Poisson arrivals + occasional bursts

Usage:
    export TRAFFIC_RATE_PER_MIN=20
    export BURST_EVERY=120
    export BURST_SIZE=5
    export MAX_EVENTS=200  # 0 for infinite
    export PII_MAX_PER_MIN=2
    python scripts/generate_canary_traffic.py
"""

import math
import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, "src")

from jeffrey.core.emotion_backend import ProtoEmotionDetector

# ============================================================================
# CONFIGURATION (via env vars)
# ============================================================================
RATE_PER_MIN = float(os.getenv("TRAFFIC_RATE_PER_MIN", "20"))
BURST_EVERY = int(os.getenv("BURST_EVERY", "120"))
BURST_SIZE = int(os.getenv("BURST_SIZE", "5"))
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "0"))  # 0 = infinite
JITTER_MS = int(os.getenv("JITTER_MS", "300"))
PII_MAX_PER_MIN = int(os.getenv("PII_MAX_PER_MIN", "2"))
SEED = int(os.getenv("TRAFFIC_SEED", str(int(time.time()))))

random.seed(SEED)

# ============================================================================
# PHRASE POOLS - Diversité maximale FR/EN
# ============================================================================

# --- NORMAL TRAFFIC (70%) ---

FR_HAPPY = [
    "Je suis très content aujourd'hui !",
    "Trop bien 😄",
    "Quel bonheur, vraiment.",
    "J'ai passé une super journée, merci !",
    "Je suis aux anges !",
    "Génial, exactement ce que j'attendais 🎉",
    "Tellement heureux de cette nouvelle !",
    "J'adore quand ça se passe bien comme ça !",
]

EN_HAPPY = [
    "I'm so happy!",
    "Feeling great today :)",
    "Absolutely delighted!",
    "What a wonderful day!",
    "This made my day! 😊",
    "Best news ever!",
    "I'm over the moon!",
    "Couldn't be happier about this!",
]

FR_SAD = [
    "Je suis triste, je n'ai pas le moral.",
    "Ça me déprime un peu…",
    "J'ai la gorge serrée.",
    "Vraiment déçu de la tournure des événements.",
    "Je me sens un peu down aujourd'hui.",
    "Ça me fend le cœur 💔",
]

EN_SAD = [
    "I feel sad.",
    "This is heartbreaking.",
    "Not feeling well emotionally.",
    "Really disappointed with how things turned out.",
    "Feeling blue today.",
    "This hurts so much.",
]

FR_FEAR = [
    "J'ai peur que ça tourne mal.",
    "Ça m'angoisse un peu.",
    "Je suis vraiment inquiet pour la suite.",
    "J'ai des appréhensions sur ce qui va se passer.",
    "Ça me stresse énormément.",
]

EN_FEAR = [
    "I'm worried about this.",
    "A bit anxious about the outcome.",
    "Really scared about what might happen.",
    "This is making me nervous.",
    "I have concerns about the future.",
]

FR_SURPRISE = [
    "Quelle surprise !",
    "Je ne m'y attendais pas du tout.",
    "Wow, je suis choqué !",
    "Incroyable, je n'aurais jamais cru ça !",
    "C'est inattendu 😮",
]

EN_SURPRISE = [
    "What a surprise!",
    "Didn't see that coming!",
    "Wow, I'm shocked!",
    "Never expected this!",
    "That's unexpected!",
]

FR_DISGUST = [
    "C'est dégoûtant.",
    "Beurk, écœurant.",
    "Je trouve ça vraiment répugnant.",
    "Ça me soulève le cœur 🤢",
    "C'est immonde.",
]

EN_DISGUST = [
    "That's disgusting.",
    "Gross. Ugh.",
    "I find this repulsive.",
    "This makes me sick.",
    "Absolutely revolting.",
]

NEUTRAL = [
    "OK.",
    "D'accord.",
    "Noted.",
    "Je m'en fiche un peu.",
    "Whatever.",
    "It's fine.",
    "Entendu.",
    "Pas de souci.",
    "C'est comme tu veux.",
    "Ça me va.",
]

# --- BORDERLINE TRAFFIC (20%) - Stress tie-break & low confidence ---

FR_ANGER = [
    "Je suis **en rage** contre ce service !",
    "Ça m'énerve vraiment !",
    "Je suis tellement en colère, hors de moi.",
    "Franchement, c'est abusé.",
    "Je suis furieux de cette situation !",
    "Ça me met hors de moi, sérieux.",
    "Je rage à fond là !",
    "C'est inadmissible, je suis révolté !",
    "Je bouillonne de colère.",
    "Je suis énervé au plus haut point !",
]

EN_ANGER = [
    "I'm so angry about this!",
    "Fuming right now.",
    "Absolutely furious.",
    "This is outrageous.",
    "I'm livid!",
    "This makes me so mad!",
    "I'm enraged by this behavior.",
    "Completely infuriated.",
    "This pisses me off!",
    "I'm seeing red right now.",
]

FR_FRUSTR = [
    "C'est frustrant cette situation.",
    "Je suis agacé mais j'essaie de rester calme.",
    "Ça me saoule, sérieux…",
    "C'est vraiment chiant cette histoire.",
    "Je suis un peu contrarié.",
    "Ça me gonfle pas mal quand même.",
    "C'est relou, franchement.",
    "Je commence à en avoir marre.",
]

EN_FRUSTR = [
    "This is frustrating.",
    "Annoyed, but okay.",
    "I'm irritated with this behavior.",
    "Getting tired of this.",
    "This is getting on my nerves.",
    "Somewhat bothered by this.",
    "This is a bit annoying.",
    "Starting to lose patience here.",
]

BORDERLINE = [
    "So angry… or maybe just disappointed.",
    "Je suis vénère mais fatigué aussi.",
    "I'm upset, not sure if angry or just annoyed.",
    "C'est pas cool, je suis pas bien mais je sais pas si c'est de la colère.",
    "Frustré ou en colère ? Je sais plus trop.",
    "I feel both happy and worried at the same time.",
    "Content mais inquiet quand même.",
    "Surprised but also a bit scared.",
    "Dégouté mais aussi triste de cette situation.",
    "Am I sad or just disappointed? Hard to tell.",
    "C'est entre la joie et la surprise, difficile à dire.",
    "Feeling a mix of anger and sadness right now.",
]

# --- PII CASES (5%) - Validation redaction ---

PII_CASES = [
    "Contacte-moi à jean.dupont@example.com",
    "Mon email pro : marie.martin@company.fr",
    "Écris-moi sur john.doe@gmail.com",
    "Va sur https://example.com/secret",
    "Regarde cette page : http://www.test.com/data",
    "Mon site : https://monsite.fr/contact",
    "Mon numéro : +33 6 12 34 56 78",
    "Appelle-moi au 06.78.90.12.34",
    "Tel: +33-1-42-68-53-00",
    "Server IP: 192.168.1.1",
    "IP du serveur : 10.0.0.5",
    "Contact: david@jeffrey-ai.com et tel +33612345678",
    "IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
]

# --- EDGE CASES (5%) - Stress robustness ---

EDGE_CASES = [
    "OK.",
    "?",
    "Bof",
    "Mdr",
    "Lol",
    "😡",
    "😊😊😊",
    "🤔",
    "!!!",
    "...",
    "Super, encore un bug... génial vraiment.",  # sarcasme
    "Wahouuuu c tro biiiien !!!",  # typos intentionnels
    "JE SUIS VRAIMENT PAS CONTENT LÀ",  # ALL CAPS
    "jsui vener mec srx",  # slang + typos
    "I'm happy mais worried en même temps tu vois",  # Franglais
    (
        "C'est vraiment vraiment vraiment vraiment vraiment frustrant cette situation qui n'en finit plus et qui me met dans un "
        "état d'énervement grandissant de jour en jour sans que rien ne change jamais."
    ),  # Très long (200+ chars)
    "wtf is this even",  # slang EN
    "omg noooo 😱😱😱",
    "meh idc tbh",  # acronymes
    "",  # empty string edge case
    "   ",  # whitespace only
]

# ============================================================================
# WEIGHTED POOLS STRUCTURE
# ============================================================================

POOLS = {
    "normal": (
        FR_HAPPY
        + EN_HAPPY
        + FR_SAD
        + EN_SAD
        + FR_FEAR
        + EN_FEAR
        + FR_SURPRISE
        + EN_SURPRISE
        + FR_DISGUST
        + EN_DISGUST
        + NEUTRAL
    ),
    "borderline": BORDERLINE + FR_ANGER + EN_ANGER + FR_FRUSTR + EN_FRUSTR,
    "pii": PII_CASES,
    "edge": EDGE_CASES,
}

WEIGHTS = {"normal": 0.70, "borderline": 0.20, "pii": 0.05, "edge": 0.05}

# PII rate limiter (sliding window)
_pii_timestamps = deque()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def mutate(text: str) -> str:
    """Introduce small random noise to avoid duplicates & stress encoder."""
    if not text or not text.strip():
        return text

    suffixes = ["", " !!", "…", "?!", " :)", " :(", " 😑", " 😡", " 😅", " #help", " please"]
    t = text

    # Random spacing tweaks
    if random.random() < 0.25:
        t = t.replace("  ", " ")

    # Random casing tweaks
    if random.random() < 0.15:
        t = t.capitalize()

    # Random suffix
    if random.random() < 0.15:
        t = t + random.choice(suffixes)

    return t


def poisson_sleep(rate_per_min: float) -> float:
    """Poisson process inter-arrival times (exponential distribution)."""
    if rate_per_min <= 0:
        return 1.0

    lam = rate_per_min / 60.0  # events per second
    u = random.random()
    delay = -math.log(1 - u) / lam

    # Add jitter
    delay += random.uniform(-JITTER_MS / 1000.0, JITTER_MS / 1000.0)

    return max(0.05, delay)


def pick_text() -> str:
    """
    Pick text from weighted pools with PII rate limiting.

    Distribution:
    - 70% normal traffic
    - 20% borderline (anger/frustration mix, ambiguous)
    - 5% PII (capped at PII_MAX_PER_MIN)
    - 5% edge cases
    """
    pools = list(POOLS.keys())
    weights = [WEIGHTS[k] for k in pools]
    k = random.choices(pools, weights=weights, k=1)[0]

    # Enforce PII rate limit
    if k == "pii":
        now = time.time()
        # Clean old timestamps (> 60s)
        while _pii_timestamps and now - _pii_timestamps[0] > 60:
            _pii_timestamps.popleft()

        # Check if cap reached
        if len(_pii_timestamps) >= PII_MAX_PER_MIN:
            # Fallback to normal if PII cap reached
            k = "normal"
        else:
            _pii_timestamps.append(now)

    return mutate(random.choice(POOLS[k]))


# ============================================================================
# MAIN TRAFFIC GENERATOR
# ============================================================================


def main():
    """Generate continuous traffic with monitoring."""

    print("=" * 70)
    print("🚀 JEFFREY OS - CANARY TRAFFIC GENERATOR")
    print("=" * 70)
    print("📊 Configuration:")
    print(f"   Rate: {RATE_PER_MIN} events/min")
    print(f"   Burst: {BURST_SIZE} events every {BURST_EVERY}s")
    print(f"   PII limit: {PII_MAX_PER_MIN}/min")
    print(f"   Max events: {MAX_EVENTS if MAX_EVENTS > 0 else 'infinite'}")
    print(f"   Seed: {SEED}")
    print(
        f"   Weights: normal={WEIGHTS['normal']:.0%}, borderline={WEIGHTS['borderline']:.0%}, "
        f"PII={WEIGHTS['pii']:.0%}, edge={WEIGHTS['edge']:.0%}"
    )
    print("=" * 70)
    print("🛑 Press Ctrl+C to stop")
    print()

    detector = ProtoEmotionDetector()
    n = 0
    last_burst = time.time()
    start_time = time.time()

    try:
        while True:
            n += 1
            text = pick_text()

            # Send prediction (monitoring called inside backend)
            try:
                res = detector.predict_proba(text)

                # Echo every 20 events
                if n % 20 == 0:
                    try:
                        # Handle dict or object response
                        if isinstance(res, dict):
                            emo = res.get("primary", "?")
                            conf = res.get("confidence", 0.0)
                        else:
                            emo = getattr(res, "emotion", getattr(res, "primary", "?"))
                            conf = getattr(res, "confidence", 0.0)

                        elapsed = time.time() - start_time
                        rate_actual = (n / elapsed) * 60 if elapsed > 0 else 0

                        print(f"#{n:05d} · '{text[:42]}…' → {emo} ({conf:.3f}) | Rate: {rate_actual:.1f}/min")
                    except Exception as e:
                        print(f"#{n:05d} · sent (echo error: {e})")

            except Exception as e:
                print(f"⚠️  Error on event #{n}: {e}")
                continue

            # Burst logic
            now = time.time()
            if now - last_burst > BURST_EVERY:
                print(f"🔥 Burst +{BURST_SIZE} events")
                for _ in range(BURST_SIZE):
                    try:
                        burst_text = pick_text()
                        detector.predict_proba(burst_text)
                    except Exception as e:
                        print(f"⚠️  Burst error: {e}")
                        continue
                last_burst = now

            # Check max events
            if MAX_EVENTS and n >= MAX_EVENTS:
                break

            # Poisson sleep
            time.sleep(poisson_sleep(RATE_PER_MIN))

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")

    finally:
        elapsed = time.time() - start_time
        rate_avg = (n / elapsed) * 60 if elapsed > 0 else 0
        print("=" * 70)
        print("✅ Traffic generation complete")
        print(f"   Events generated: {n}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average rate: {rate_avg:.1f} events/min")
        print("=" * 70)


if __name__ == "__main__":
    main()
