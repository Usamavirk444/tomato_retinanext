# Audit history

Files kept for provenance. **Nothing here is read by any script.** They are
superseded or disputed artefacts, retained so the record is complete.

| File | Why it is here |
|---|---|
| `main_results_table_SUPERSEDED_2026-05.csv` | May 2026 results table. Gives YOLOv8n 90.17 and RetinaNeXt 93.33, both superseded by the October table in `reference_results/in_domain/main_results_table.csv` (87.65 / 94.42), which the manuscript uses. Kept so the discrepancy is visible rather than lost. |

The disputed S3 seed-456 September checkpoint is not stored here — it is a
434 MB binary. Its hash and provenance are recorded in
`manifests/checkpoint_inventory.csv` and `docs/S3_SEED456_INVESTIGATION.md`.
