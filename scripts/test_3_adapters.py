import asyncio
import warnings

# Warnings async bloquants
warnings.simplefilter("error", RuntimeWarning)


async def test():
    """Test des 3 adaptateurs réels."""
    print("=== TEST ADAPTATEURS RÉELS ===")

    # Test EmotionAdapter
    try:
        from jeffrey.bridge.adapters.emotion_adapter import EmotionAdapter

        e = EmotionAdapter()
        result = await e.process({"text": "test"})
        print(f"✅ EmotionAdapter OK: {result.get('status', 'unknown')}")
    except Exception as ex:
        print(f"❌ EmotionAdapter: {ex}")
        return False

    # Test ExecutiveAdapter
    try:
        from jeffrey.bridge.adapters.executive_adapter import ExecutiveAdapter

        ex = ExecutiveAdapter()
        result = await ex.process({"input": "test"})
        print(f"⚠️ ExecutiveAdapter: {result.get('status', 'unknown')} - {result.get('error', 'OK')}")
    except Exception as e:
        print(f"⚠️ ExecutiveAdapter: {e}")

    # Test MotorAdapter
    try:
        from jeffrey.bridge.adapters.motor_adapter import MotorAdapter

        m = MotorAdapter()
        result = await m.process({"input": "test"})
        print(f"⚠️ MotorAdapter: {result.get('status', 'unknown')} - {result.get('error', 'OK')}")
    except Exception as e:
        print(f"⚠️ MotorAdapter: {e}")

    print("=== FIN TEST ===")
    return True


if __name__ == "__main__":
    asyncio.run(test())
