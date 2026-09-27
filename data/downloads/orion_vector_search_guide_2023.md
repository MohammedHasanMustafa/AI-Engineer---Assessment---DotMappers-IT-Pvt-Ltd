# Orion Vector Search Platform: Engineering Guide (2023 edition)

Orion is the internal semantic search platform used by our product teams to build retrieval-augmented generation (RAG) features. This guide records the configuration defaults that the platform team recommended in 2023.

## Chunking

The recommended chunk size for Orion ingestion is 512 tokens with an overlap of 50 tokens between consecutive chunks. Larger chunks keep more surrounding context in each passage, which our 2023 evaluation found helpful for long technical manuals. Chunks smaller than 128 tokens should be merged with their neighbours.

## Retrieval defaults

The default top-k for Orion retrieval should be 10 passages. Returning ten passages maximises recall and lets the generator decide which passages are useful. Retrieval uses dense vectors only; keyword search was evaluated and not adopted.

## Reranking

In our 2023 benchmarks, adding a cross-encoder reranker provided no measurable improvement in precision at 5 and added about 120 milliseconds of latency per query. We therefore recommend against enabling reranking on Orion.

## Index configuration

Orion uses an HNSW index with M set to 16 and efSearch set to 64. Indexes are rebuilt nightly. Embeddings are L2-normalized and compared with inner product.

## Monitoring

Teams should track retrieval latency at the 95th percentile and the share of queries that return no passage above the relevance threshold. Alerts fire when the 95th percentile latency exceeds 300 milliseconds.
