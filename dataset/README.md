# Dataset format

ChipMEM expects one directory per task. Each task directory contains the exact UTF-8 artifact declared by `task_artifact` in the experiment configuration.

```text
dataset/
└── <dataset-name>/
    ├── task_a/
    │   └── TASK.md
    └── task_b/
        └── TASK.md
```

Only the declared task artifact is embedded for retrieval. Do not add agent instructions, tool syntax, generated files, evaluation artifacts, or learned skills to that artifact.

The `synthetic/` dataset is a privacy-safe executable example. It is not paper data, a benchmark result, or a proprietary dataset. Private or licensed datasets should remain outside this repository and be referenced through a local configuration path.
