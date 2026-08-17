# G1 commanded locomotion v1 artifact

This directory contains the single selected training checkpoint and compact
evidence retained for the public release.

| File | Purpose |
| --- | --- |
| `model_37300.pt` | selected PPO checkpoint for Play/resume/audit |
| `agent.yaml` | final runner and PPO configuration snapshot |
| `env.yaml` | final task, observation, command, reward, and terrain snapshot |
| `training_summary.json` | scalar summaries from iterations 32300–37300 |
| `MANIFEST.sha256` | integrity manifest |

The matching deployment actor is stored at
`deploy/robots/g1/config/policy/velocity/v1/exported/policy.onnx`; it is
included in the same manifest.

Verify from the repository root:

~~~bash
sha256sum --check artifacts/g1-commanded-locomotion-v1/MANIFEST.sha256
~~~

The original TensorBoard event, optimizer history outside the selected PT,
intermediate checkpoints, crash dumps, and host-specific paths are intentionally
excluded. See `docs/model-card.md` for the contract, metrics, intended use,
and limitations.
