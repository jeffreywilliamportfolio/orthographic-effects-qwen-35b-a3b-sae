# Push Checklist

Target repository:

`https://github.com/jeffreywilliamportfolio/orthographic-effects-qwen-35b-a3b-sae`

Before pushing:

- Confirm `git status --short` only includes the intended publish files.
- Confirm no `.env`, token file, private key, model weight, SAE checkpoint, `.pt`, or tar archive is staged.
- Confirm generated plots are present under `results/full_controlled_perturbation_matrix/aggregates/plots/`.
- Confirm `RESULTS.md` and `PROVENANCE.md` match the intended public claims.

Push command after remote repo exists:

```sh
git push -u origin main
```

