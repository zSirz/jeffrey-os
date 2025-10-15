# Audit data distribution script for Phase 2 Sprint 1
import yaml

with open('tests/data/bootstrap_seed.yaml') as f:
    data = yaml.safe_load(f)

emotions = {}
for item in data:
    emo = item['emotion']
    emotions[emo] = emotions.get(emo, 0) + 1

print('\n📊 Distribution classes :')
for k, v in sorted(emotions.items(), key=lambda x: x[1]):
    print(f'  {k:12s}: {v:3d}')
print(f'\nTotal: {sum(emotions.values())}')

# Check frustration
if emotions.get('frustration', 0) < 25:
    print('\n⚠️  ATTENTION : frustration < 25 → Augmentation recommandée')
else:
    print('\n✅ frustration ≥ 25 → OK pour fine-tuning')
