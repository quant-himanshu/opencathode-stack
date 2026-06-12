#!/bin/bash
# Extract VED dynamic data from 7z archives after download completes.
# Run: bash scripts/extract_ved.sh

DATA_DIR="$(dirname "$0")/../data/ved"
ARCH_DIR="${DATA_DIR}/Data"

for part in 1 2; do
    ARCHIVE="${ARCH_DIR}/VED_DynamicData_Part${part}.7z"
    if [ ! -f "$ARCHIVE" ]; then
        echo "ERROR: $ARCHIVE not found. Download it first."
        exit 1
    fi
    SIZE=$(stat -f%z "$ARCHIVE" 2>/dev/null || stat -c%s "$ARCHIVE")
    echo "Extracting ${ARCHIVE} (${SIZE} bytes)..."
    /opt/homebrew/bin/7za x "$ARCHIVE" -o"${DATA_DIR}" -y
    echo "Done: Part ${part}"
done

echo ""
echo "Extraction complete. Weekly CSV files are in: ${DATA_DIR}"
ls -lh "${DATA_DIR}"/*.csv 2>/dev/null | head -10 || echo "(no CSVs yet - extraction may still be running)"
