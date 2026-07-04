# Émulateur d'ordonnancement

L'émulateur sert à trois choses:

- comparer plusieurs politiques d'ordonnancement sur le même workload;
- générer des états de décision pour entraîner le modèle candidat;
- rejouer un modèle entraîné en `shadow` ou en `closed-loop` contre `CFS`.

## Lancer une simulation simple

```bash
python3 emulator/sched_em.py emulator/tasks/demo.tasks -p CFS -d 220 --seed 3
```

Comparaison multi-politiques:

```bash
python3 emulator/sched_em.py emulator/tasks/demo.tasks --compare --seed 3
```

## Générer un dataset pour le modèle

Le chemin recommandé passe par `LSTM/scripts/build_candidate_dataset.py`.

Pour créer beaucoup de workloads variés avant ça:

```bash
python3 emulator/generate_workloads.py \
    --output-dir emulator/tasks/generated \
    --count 48 \
    --seed 42
```

Exemple:

```bash
cd LSTM
uv run python scripts/build_candidate_dataset.py \
    --taskfiles ../emulator/tasks/demo.tasks ../emulator/tasks/all_in_one.tasks \
    --output-dir artifacts/candidate_dataset \
    --policies CFS \
    --seeds 0 1 2 3 \
    --duration 220 \
    --teacher cfs \
    --max-history 256
```

## Rejouer un modèle entraîné

Le modèle ne prédit plus un nom de tâche global. Il reçoit:

- les candidats runnable;
- l'état de toutes les tâches;
- les `256` dernières décisions.

Puis il choisit directement un candidat.

### Shadow

Le modèle prédit, mais `CFS` garde la main. L'émulateur mesure ce qu'aurait changé une divergence.

```bash
uv run --project LSTM python emulator/sched_em.py emulator/tasks/demo.tasks \
    -p CFS \
    -d 220 \
    --seed 3 \
    --model-mode shadow \
    --model-run-dir LSTM/artifacts/training_runs/candidate_baseline \
    --stats-json emulator/artifacts/benchmarks/demo_shadow.json
```

### Closed-loop

Le modèle choisit vraiment la prochaine tâche parmi les candidats. Si la base de comparaison est `CFS`, l'émulateur exporte aussi les deltas contre `CFS`.

```bash
uv run --project LSTM python emulator/sched_em.py emulator/tasks/demo.tasks \
    -p CFS \
    -d 220 \
    --seed 3 \
    --model-mode closed-loop \
    --model-run-dir LSTM/artifacts/training_runs/candidate_baseline \
    --stats-json emulator/artifacts/benchmarks/demo_closed_loop.json
```

Les métriques suivies sont:

- `avg_wait_ms`
- `avg_turnaround_ms`
- `avg_response_ms`
- `fairness_index`
- `ctx_switches`
- `starvation_count`

## Traces synthétiques

L'émulateur peut aussi exporter une trace `sched_switch` synthétique pour inspection:

```bash
python3 emulator/sched_em.py emulator/tasks/demo.tasks \
    -p CFS \
    -d 120 \
    --seed 7 \
    --cpu-id 0 \
    --trace-out emulator/artifacts/synthetic/demo.trace
```

Cette exportation reste utile pour déboguer les décisions et conserver une trace temporelle lisible.
