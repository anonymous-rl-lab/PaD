# Bundled Python distribution

`pad_causal-0.1.1-py3-none-any.whl` is the built `pad-causal` package the
manuscript refers to. It is bundled here so that reviewers can install the
exact reviewed build without visiting a package index or a named account.

```bash
python -m pip install ./tools/pad_causal-0.1.1-py3-none-any.whl
pad-causal demo --output pad_demo
```

SHA-256 of the wheel:

```
78537382c4756926285a8265c0d971b02d7b8c84340065c55a56980f11df200f
```

The wheel carries no author, maintainer or contact metadata. Python 3.10 or
newer is required; runtime dependencies are NumPy, SciPy and scikit-learn.
The frozen research code in `src/pad/` remains the reference implementation
that `reproduce.py` executes; the wheel is the reusable API and command-line
packaging of the same algorithm.
