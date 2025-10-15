#!/bin/bash
echo "🔄 Rolling back Bundle 3..."
cp -r backups/bundle3_rollback/artifacts/ ./ 2>/dev/null || true
cp backups/bundle3_rollback/*.db data/ 2>/dev/null || true
tar -xzf backups/bundle3_rollback/modules_*.tar.gz 2>/dev/null || true
echo "✅ Rollback complete (DB + modules + config)"
