# Contributing to metad-llm

Thanks for your interest in improving metad-llm. Bug reports, ideas, documentation fixes and code are all welcome.

## Getting set up

```bash
git clone https://github.com/kishoren18-hue/metad-llm
cd metad-llm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install        # optional: lint on every commit
```

Run the checks CI runs:

```bash
ruff check .
pytest --cov=metad_llm
```

## Making a change

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Create a branch, make the change, and add or update tests.
3. Keep pull requests focused on one thing; describe what changed and why.
4. Make sure the checks above pass, then open a pull request.

Good tests matter more than clever code here. A bug fix should come with a test that fails without the fix.

## Ideas to start with

* Add the unequal-variance SDT model (s != 1)
* Add response-specific meta-d' (separate fits for S1 and S2 responses)
* A Colab notebook that evaluates an open model end to end

## Code of conduct

By participating you agree to follow the [code of conduct](CODE_OF_CONDUCT.md).
