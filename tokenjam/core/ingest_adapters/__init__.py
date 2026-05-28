"""
Ingest adapters — inbound. Read external systems and produce TokenJam
NormalizedSpan records, written to the local DB via the standard ingest
pipeline (or via direct upsert when the pipeline isn't available).

This package is the canonical home for adapters that ingest from other
observability tools (Langfuse, Helicone, raw OTLP). Each adapter exposes
a CLI subcommand under `tj backfill <source>`.

Distinct from `tokenjam.sdk.integrations/`, which is outbound — those
monkey-patch the user's agent runtime to emit OTel spans into tj.
"""
