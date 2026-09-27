# Orion Vector Search Platform: Engineering Guide (2024 revision)

This revision replaces several recommendations of the 2023 guide after a new evaluation on 4,000 labelled queries from production traffic.

## Chunking

The recommended chunk size for Orion ingestion is 256 tokens with a 20 percent overlap, which is about 50 tokens. Smaller chunks produced more precise citations and reduced the amount of irrelevant text passed to the generator. Chunks smaller than 64 tokens should be merged with their neighbours.

## Retrieval defaults

The default top-k for Orion retrieval should be 3 passages, selected after reranking a candidate pool of 25 passages. Sending fewer passages to the generator lowered hallucination rates in our evaluation. Retrieval now combines dense vectors with BM25 keyword search using reciprocal rank fusion.

## Reranking

Cross-encoder reranking improved precision at 5 by 18 percent on the 2024 evaluation set, at a cost of about 90 milliseconds of added latency. Reranking is now enabled by default on Orion.

## Index configuration

Orion continues to use an HNSW index with M set to 16; efSearch was raised to 128 to improve recall. Embeddings are L2-normalized and compared with inner product.

## Monitoring

Alerts fire when the 95th percentile retrieval latency exceeds 250 milliseconds, or when more than 15 percent of queries in an hour are refused for insufficient evidence.
