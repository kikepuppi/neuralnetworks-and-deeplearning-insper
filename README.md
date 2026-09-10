# ANN & DL (Insper 2026.2)

Portfolio for Artificial Neural Networks and Deep Learning, published with MkDocs
Material on GitHub Pages. English at the root, Portuguese under `/pt/`.

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
    data/       index.md + index.pt.md + code/ + figures/ + data/
    perceptron/ index.md + index.pt.md + code/ + figures/
    mlp/        index.md + index.pt.md + code/ + figures/
  projects/
    index.md
    eda/            index.md + code/ + figures/
    classification/ index.md + code/ + figures/
    regression/     index.md + code/ + figures/
    generative/     index.md + code/ + figures/
mkdocs.yml
requirements.txt
```

Files named `index.md` are English, the default language. Files named `index.pt.md` are
the Portuguese translation. A page missing in one language falls back to the other.

## Publishing

`.github/workflows/pages.yml` builds and deploys on every push to `main`.
Set *Settings > Pages > Source* to *GitHub Actions*.
