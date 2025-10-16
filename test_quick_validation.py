import asyncio
import httpx
import json

async def quick_test():
    async with httpx.AsyncClient() as client:
        # Test 5 émotions rapides
        for i in range(5):
            resp = await client.post(
                "http://localhost:8000/api/v1/emotion/detect",
                json={"text": f"Test emotion {i}"},
                timeout=3.0
            )
            print(f"Request {i}: {resp.status_code}")
        
        # Attendre un peu
        await asyncio.sleep(2)
        
        # Vérifier le fortress status
        fortress = await client.get("http://localhost:8000/api/v1/brain/fortress")
        data = fortress.json()
        
        memories = data['pipeline']['memories_stored']
        thoughts = data['pipeline']['thoughts_generated']
        ratio = thoughts / max(memories, 1)
        
        print(f"\nRésultats:")
        print(f"  Memories: {memories}")
        print(f"  Thoughts: {thoughts}")
        print(f"  Ratio: {ratio:.2f}")
        print(f"  Queue: {data['event_bus'].get('queue_size', 'N/A')}")
        
        if ratio > 0.5:
            print("✅ BUG IDEMPOTENCE FIXÉ!")
        else:
            print("❌ Ratio toujours faible")

asyncio.run(quick_test())
