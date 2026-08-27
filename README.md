# ANN & DL (Insper 2026.2)

Exercise portfolio for Artificial Neural Networks and Deep Learning, published with
MkDocs Material on GitHub Pages. The site is bilingual: English at the root, Portuguese
under `/pt/`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdocs serve
```

## Layout

```
docs/
  index.md / index.pt.md
  exercises/
    data/       index.md + index.pt.md + code/ + figures/
    perceptron/ index.md + index.pt.md + code/ + figures/
    mlp/        index.md + index.pt.md + code/ + figures/
  projects/index.md
mkdocs.yml
requirements.txt
```

Untagged files (`index.md`) are English, the default language. Files suffixed `.pt.md`
are the Portuguese translation. A page missing in one language falls back to the other.

## Publishing

`.github/workflows/pages.yml` builds and deploys on every push to `main`.
Enable it under *Settings > Pages > Source: GitHub Actions*.
