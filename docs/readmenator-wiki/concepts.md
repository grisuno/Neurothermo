# Concepts

Second-brain semantic layer: nouns map atomically to file sets (EXTRACTED); verbs aggregate structural edges (INFERRED).

| Concept | Files | Mentions | Top Files |
|---------|-------|----------|-----------|
| `metrics` | 2 | 10 | `app.py`, `neurothermo.py` |
| `delta` | 2 | 8 | `app.py`, `neurothermo.py` |
| `only` | 2 | 8 | `app.py`, `neurothermo.py` |
| `all` | 2 | 6 | `app.py`, `neurothermo.py` |
| `summary` | 2 | 6 | `app.py`, `neurothermo.py` |
| `training` | 2 | 6 | `app.py`, `neurothermo.py` |
| `during` | 2 | 5 | `app.py`, `neurothermo.py` |
| `neurothermo` | 2 | 5 | `app.py`, `neurothermo.py` |
| `computed` | 2 | 3 | `app.py`, `neurothermo.py` |
| `history` | 2 | 3 | `app.py`, `neurothermo.py` |
| `instant` | 2 | 2 | `app.py`, `neurothermo.py` |
| `library` | 2 | 2 | `app.py`, `neurothermo.py` |

## Verb Edges

| Source | Verb | Target | Strength |
|--------|------|--------|----------|
| `all` | `depends_on` | `computed` | 1.00 |
| `all` | `depends_on` | `delta` | 1.00 |
| `all` | `depends_on` | `during` | 1.00 |
| `all` | `depends_on` | `history` | 1.00 |
| `all` | `depends_on` | `instant` | 1.00 |
| `all` | `depends_on` | `library` | 1.00 |
| `all` | `depends_on` | `metrics` | 1.00 |
| `all` | `depends_on` | `neurothermo` | 1.00 |
| `all` | `depends_on` | `only` | 1.00 |
| `all` | `depends_on` | `summary` | 1.00 |
| `all` | `depends_on` | `training` | 1.00 |
| `computed` | `depends_on` | `all` | 1.00 |
| `computed` | `depends_on` | `delta` | 1.00 |
| `computed` | `depends_on` | `during` | 1.00 |
| `computed` | `depends_on` | `history` | 1.00 |
| `computed` | `depends_on` | `instant` | 1.00 |
| `computed` | `depends_on` | `library` | 1.00 |
| `computed` | `depends_on` | `metrics` | 1.00 |
| `computed` | `depends_on` | `neurothermo` | 1.00 |
| `computed` | `depends_on` | `only` | 1.00 |
| `computed` | `depends_on` | `summary` | 1.00 |
| `computed` | `depends_on` | `training` | 1.00 |
| `delta` | `depends_on` | `all` | 1.00 |
| `delta` | `depends_on` | `computed` | 1.00 |
| `delta` | `depends_on` | `during` | 1.00 |
| `delta` | `depends_on` | `history` | 1.00 |
| `delta` | `depends_on` | `instant` | 1.00 |
| `delta` | `depends_on` | `library` | 1.00 |
| `delta` | `depends_on` | `metrics` | 1.00 |
| `delta` | `depends_on` | `neurothermo` | 1.00 |
| `delta` | `depends_on` | `only` | 1.00 |
| `delta` | `depends_on` | `summary` | 1.00 |
| `delta` | `depends_on` | `training` | 1.00 |
| `during` | `depends_on` | `all` | 1.00 |
| `during` | `depends_on` | `computed` | 1.00 |
| `during` | `depends_on` | `delta` | 1.00 |
| `during` | `depends_on` | `history` | 1.00 |
| `during` | `depends_on` | `instant` | 1.00 |
| `during` | `depends_on` | `library` | 1.00 |
| `during` | `depends_on` | `metrics` | 1.00 |
| `during` | `depends_on` | `neurothermo` | 1.00 |
| `during` | `depends_on` | `only` | 1.00 |
| `during` | `depends_on` | `summary` | 1.00 |
| `during` | `depends_on` | `training` | 1.00 |
| `history` | `depends_on` | `all` | 1.00 |
| `history` | `depends_on` | `computed` | 1.00 |
| `history` | `depends_on` | `delta` | 1.00 |
| `history` | `depends_on` | `during` | 1.00 |
| `history` | `depends_on` | `instant` | 1.00 |
| `history` | `depends_on` | `library` | 1.00 |

## Dialectic Prompts

- Thesis: `all` centralizes 2 files; Antithesis: `computed` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `delta` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `during` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `history` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `instant` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `library` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `metrics` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `neurothermo` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `only` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `all` centralizes 2 files; Antithesis: `summary` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
