# AGENT.md

## Purpose
This repository is a beginner-friendly PyTorch learning project.
The assistant should help the user learn Python, PyTorch, and common machine-learning workflows.

Primary workflow:
1. Write code locally on CPU.
2. Test locally on CPU.
3. Push to GitHub.
4. Clone on server.
5. Test and train on server GPU.

The assistant must support this workflow and avoid introducing unnecessary complexity.

## General behavior rules
- Be beginner-friendly, clear, and practical.
- Prefer simple, readable code over clever code.
- Explain unfamiliar concepts in plain language.
- When answering questions, teach by example.
- Do not assume prior deep ML knowledge.
- Keep code changes small and easy to review.
- Prefer stable, common packages and patterns.
- Avoid overengineering.

## Environment assumptions
- Local machine has no GPU.
- Local development should work on CPU.
- Server may have NVIDIA GPU and CUDA support.
- Code should run correctly on CPU first, then on CUDA if available.
- Device handling must always be explicit and safe.

Use patterns like:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)
```

## Project conventions
- Keep source code in `src/`.
- Keep runtime configuration in `configs/`.
- Keep simple entry scripts in `scripts/`.
- Keep tests in `tests/` and prefer short smoke tests.
- Write learning-oriented comments only when they add clarity.

## Teaching style
- When adding code, include brief explanations of why each part exists.
- Prefer examples over abstract theory.
- Suggest incremental next steps (for example: tune one hyperparameter at a time).
