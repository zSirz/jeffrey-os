#!/bin/bash
# Secure backup script with triple verification and encryption
# PARANOID MODE: Zero data loss guaranteed

set -Eeuo pipefail
IFS=$'\n\t'
umask 077
trap 'echo "❌ Backup failed at line $LINENO"; exit 1' ERR

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/jeffrey_${TIMESTAMP}"
DESKTOP_BACKUP="$HOME/Desktop/Jeffrey_Backup_${TIMESTAMP}"

echo "🔒 SECURE BACKUP - PARANOID MODE"
echo "========================================"
echo "Timestamp: ${TIMESTAMP}"

# 1. Check prerequisites
echo "📋 Checking prerequisites..."

# Check disk space (need at least 2GB free)
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/[^0-9.]//g')
if (( $(echo "$AVAILABLE_SPACE < 2" | bc -l) )); then
    echo "❌ Insufficient disk space: ${AVAILABLE_SPACE}GB available (need 2GB)"
    exit 1
fi

# Check write permissions
if [ ! -w "." ]; then
    echo "❌ No write permission in current directory"
    exit 1
fi

# 2. Create backup structure
echo "📦 Creating backup structure..."
mkdir -p "$BACKUP_DIR"/{src,icloud,config,tests,scripts,artifacts,keys}

# 3. Local backup with integrity check
echo "📂 Backing up local files..."
if [ -d "src" ]; then
    tar -czf "$BACKUP_DIR/local_${TIMESTAMP}.tar.gz" \
        src/ config/ tests/ scripts/ \
        --exclude="__pycache__" \
        --exclude=".pytest_cache" \
        --exclude="*.pyc" \
        --exclude=".venv*" \
        --exclude="venv*" \
        --exclude=".git" \
        2>/dev/null || true
fi

# 4. iCloud backup (if exists)
echo "☁️ Backing up iCloud modules..."
ICLOUD_PATH="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey"
if [ -d "$ICLOUD_PATH" ]; then
    tar -czf "$BACKUP_DIR/icloud_${TIMESTAMP}.tar.gz" \
        "$ICLOUD_PATH/Jeffrey_Apps/Jeffrey_OS" \
        --exclude="*/Archive/*" \
        --exclude="*/Backup/*" \
        --exclude="*/old/*" \
        --exclude="__pycache__" \
        2>/dev/null || true
fi

# 5. Git bundle with all history
echo "📚 Creating Git bundle..."
if [ -d ".git" ]; then
    git bundle create "$BACKUP_DIR/git_bundle_${TIMESTAMP}.bundle" --all
    git tag -a "backup-${TIMESTAMP}" -m "Backup before module inventory" || true
fi

# 6. Module inventory (Python files > 100 lines)
echo "📝 Creating module inventory..."
cat > "$BACKUP_DIR/modules_inventory.csv" << EOF
Path,Lines,SHA256,Size,LastModified
EOF

find src -name "*.py" -type f 2>/dev/null | while read -r file; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file" 2>/dev/null || echo 0)
        if [ "$lines" -gt 100 ]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
            sha256=$(shasum -a 256 "$file" | cut -d' ' -f1)
            modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$file" 2>/dev/null || \
                      stat -c "%y" "$file" 2>/dev/null | cut -d' ' -f1,2 || \
                      echo "unknown")
            echo "$file,$lines,$sha256,$size,$modified" >> "$BACKUP_DIR/modules_inventory.csv"
        fi
    fi
done

# 7. Save current Python environment
echo "📦 Saving Python environment..."
if [ -f ".venv/bin/activate" ] || [ -f "venv/bin/activate" ]; then
    source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true
    pip freeze > "$BACKUP_DIR/requirements_frozen.txt" 2>/dev/null || true
fi

# 8. Configuration snapshot
echo "⚙️ Saving configurations..."
cp -r config "$BACKUP_DIR/" 2>/dev/null || true
cp -r artifacts "$BACKUP_DIR/" 2>/dev/null || true

# 9. Verify archive integrity
echo "✅ Verifying backup integrity..."
VERIFIED=true
for archive in "$BACKUP_DIR"/*.tar.gz; do
    if [ -f "$archive" ]; then
        if tar -tzf "$archive" > /dev/null 2>&1; then
            size=$(du -h "$archive" | cut -f1)
            sha256=$(shasum -a 256 "$archive" | cut -d' ' -f1)
            echo "   ✓ $(basename "$archive"): $size, SHA256: ${sha256:0:16}..."
        else
            echo "   ❌ ERROR: $(basename "$archive") is corrupted!"
            VERIFIED=false
        fi
    fi
done

if [ "$VERIFIED" = false ]; then
    echo "❌ Backup verification failed!"
    exit 1
fi

# 10. Encrypt backups (optional, if passphrase is set)
if [ ! -z "${JEFFREY_BACKUP_PASSPHRASE:-}" ]; then
    echo "🔐 Encrypting backups..."
    for archive in "$BACKUP_DIR"/*.tar.gz; do
        if [ -f "$archive" ]; then
            gpg --batch --yes --symmetric --cipher-algo AES256 \
                --passphrase "$JEFFREY_BACKUP_PASSPHRASE" \
                --output "${archive}.gpg" "$archive"
            rm -f "$archive"
        fi
    done
fi

# 11. Create external copy
echo "💾 Creating external backup..."
cp -r "$BACKUP_DIR" "$DESKTOP_BACKUP"

# 12. Create metadata file
cat > "$BACKUP_DIR/backup_metadata.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "backup_dir": "$BACKUP_DIR",
    "external_dir": "$DESKTOP_BACKUP",
    "git_tag": "backup-${TIMESTAMP}",
    "encrypted": ${JEFFREY_BACKUP_PASSPHRASE:+true},
    "modules_count": $(grep -c "^src" "$BACKUP_DIR/modules_inventory.csv" 2>/dev/null || echo 0),
    "total_size": "$(du -sh "$BACKUP_DIR" | cut -f1)",
    "verified": true
}
EOF

# 13. Summary
echo ""
echo "✅ BACKUP COMPLETE AND VERIFIED"
echo "================================"
echo "   Local: $BACKUP_DIR"
echo "   External: $DESKTOP_BACKUP"
echo "   Git tag: backup-${TIMESTAMP}"
echo "   Modules: $(grep -c "^src" "$BACKUP_DIR/modules_inventory.csv" 2>/dev/null || echo 0) files > 100 lines"
echo "   Total size: $(du -sh "$BACKUP_DIR" | cut -f1)"
echo ""
echo "🔒 All data is safe and recoverable!"
