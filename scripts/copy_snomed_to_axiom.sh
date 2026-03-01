#!/usr/bin/env bash
# Copy SNOMED CT RF2 files from Mac to Axiom Core.
# Only copies the extracted directory, not the zip.
# Target: /home/exx/data_sdb/ontologies/snomed/ on Axiom Core.

set -euo pipefail

SRC="/Users/rs/myCodeMac/SNOWMED_CT/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20250901T120000Z/"
DEST="exx@192.168.4.25:/home/exx/data_sdb/ontologies/snomed/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20250901T120000Z/"

echo "Syncing SNOMED CT RF2 files to Axiom Core..."
rsync -av --progress "$SRC" "$DEST"
echo "Done."
