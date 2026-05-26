# Git workflow

Branch-per-feature. Never push to `main` directly.

```bash
git checkout main
git pull origin main
git checkout -b feature/<lang>-<description>
# work...
git add <files>
git commit -m "Imperative mood description ≤72 chars"
git push -u origin feature/<lang>-<description>
# Open a PR on github.com, request review, merge after approval.
```

## Rules

1. One feature per PR.
2. No data files (>50 MB).
3. No model checkpoints.
4. No secrets (API keys, tokens).
5. No real names in code comments (we are anonymous for the paper).
6. No `git push --force` to `main`.

## Fixing common mistakes

| Situation | Fix |
|---|---|
| Forgot to add a file | `git add file && git commit --amend --no-edit && git push --force-with-lease` (own branch only) |
| Undo last commit, keep changes | `git reset --soft HEAD~1` |
| Throw away local changes | `git checkout -- file.py` |
| Wrong branch | `git stash && git checkout right-branch && git stash pop` |
| Merge conflict | Edit file, resolve `<<<<<<<` markers, `git add file && git commit` |
