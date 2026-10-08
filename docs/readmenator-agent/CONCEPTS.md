# Concepts

Nouns map atomically to file sets (EXTRACTED); verbs aggregate structural edges (INFERRED).

- `metrics` | files=2 | mentions=10 | `app.py`, `neurothermo.py`
- `delta` | files=2 | mentions=8 | `app.py`, `neurothermo.py`
- `only` | files=2 | mentions=8 | `app.py`, `neurothermo.py`
- `all` | files=2 | mentions=6 | `app.py`, `neurothermo.py`
- `summary` | files=2 | mentions=6 | `app.py`, `neurothermo.py`
- `training` | files=2 | mentions=6 | `app.py`, `neurothermo.py`
- `during` | files=2 | mentions=5 | `app.py`, `neurothermo.py`
- `neurothermo` | files=2 | mentions=5 | `app.py`, `neurothermo.py`
- `computed` | files=2 | mentions=3 | `app.py`, `neurothermo.py`
- `history` | files=2 | mentions=3 | `app.py`, `neurothermo.py`
- `instant` | files=2 | mentions=2 | `app.py`, `neurothermo.py`
- `library` | files=2 | mentions=2 | `app.py`, `neurothermo.py`

## Verb Edges

- `all` --depends_on--> `computed` (strength 1.00)
- `all` --depends_on--> `delta` (strength 1.00)
- `all` --depends_on--> `during` (strength 1.00)
- `all` --depends_on--> `history` (strength 1.00)
- `all` --depends_on--> `instant` (strength 1.00)
- `all` --depends_on--> `library` (strength 1.00)
- `all` --depends_on--> `metrics` (strength 1.00)
- `all` --depends_on--> `neurothermo` (strength 1.00)
- `all` --depends_on--> `only` (strength 1.00)
- `all` --depends_on--> `summary` (strength 1.00)
- `all` --depends_on--> `training` (strength 1.00)
- `computed` --depends_on--> `all` (strength 1.00)
- `computed` --depends_on--> `delta` (strength 1.00)
- `computed` --depends_on--> `during` (strength 1.00)
- `computed` --depends_on--> `history` (strength 1.00)
- `computed` --depends_on--> `instant` (strength 1.00)
- `computed` --depends_on--> `library` (strength 1.00)
- `computed` --depends_on--> `metrics` (strength 1.00)
- `computed` --depends_on--> `neurothermo` (strength 1.00)
- `computed` --depends_on--> `only` (strength 1.00)
- `computed` --depends_on--> `summary` (strength 1.00)
- `computed` --depends_on--> `training` (strength 1.00)
- `delta` --depends_on--> `all` (strength 1.00)
- `delta` --depends_on--> `computed` (strength 1.00)
- `delta` --depends_on--> `during` (strength 1.00)
- `delta` --depends_on--> `history` (strength 1.00)
- `delta` --depends_on--> `instant` (strength 1.00)
- `delta` --depends_on--> `library` (strength 1.00)
- `delta` --depends_on--> `metrics` (strength 1.00)
- `delta` --depends_on--> `neurothermo` (strength 1.00)
- `delta` --depends_on--> `only` (strength 1.00)
- `delta` --depends_on--> `summary` (strength 1.00)
- `delta` --depends_on--> `training` (strength 1.00)
- `during` --depends_on--> `all` (strength 1.00)
- `during` --depends_on--> `computed` (strength 1.00)
- `during` --depends_on--> `delta` (strength 1.00)
- `during` --depends_on--> `history` (strength 1.00)
- `during` --depends_on--> `instant` (strength 1.00)
- `during` --depends_on--> `library` (strength 1.00)
- `during` --depends_on--> `metrics` (strength 1.00)
- `during` --depends_on--> `neurothermo` (strength 1.00)
- `during` --depends_on--> `only` (strength 1.00)
- `during` --depends_on--> `summary` (strength 1.00)
- `during` --depends_on--> `training` (strength 1.00)
- `history` --depends_on--> `all` (strength 1.00)
- `history` --depends_on--> `computed` (strength 1.00)
- `history` --depends_on--> `delta` (strength 1.00)
- `history` --depends_on--> `during` (strength 1.00)
- `history` --depends_on--> `instant` (strength 1.00)
- `history` --depends_on--> `library` (strength 1.00)

## Dialectic

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
