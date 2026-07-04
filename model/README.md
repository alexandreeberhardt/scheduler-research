# Sélection de tâches candidates pour l'ordonnancement

Le modèle apprend une vraie décision d'ordonnancement:

- entrée: les tâches candidates runnable, l'état de toutes les tâches, et les `256` dernières décisions;
- sortie: l'indice du candidat à exécuter.
- architecture: encodeur de set avec attention sur `all_tasks`, attention candidat→historique avec récence relative, puis gating entre candidat / historique / contexte global.

Petit schéma:

```text
candidate_features -> task_encoder --------------------\
history_features   -> task_encoder -> +recency -> attn -+-> gate -> scorer
all_task_features  -> task_encoder -> set-attn ---------/
global_features    -> global_encoder -------------------/
```

Le dossier s'appelait historiquement `LSTM/` ; le modèle entraîné est un scoreur de candidats à base d'attention (les artefacts et scripts nommés `lstm_*` concernent l'ancien baseline récurrent).

## Architecture détaillée

### Entrées

| Tenseur | Forme | Description |
|---------|-------|-------------|
| `candidate_features` | [N × 34] | Features des tâches candidates runnable |
| `history_features` | [T × 34] | Séquence des T dernières tâches exécutées (T ≤ 256) |
| `all_task_features` | [M × 34] | Toutes les tâches connues du scheduler |
| `global_features` | [13] | État global système |

**Features par tâche (34 dims) :** PID log, état one-hot (new/ready/running/sleeping/dead), politique (CFS/FIFO/RR/batch/idle/nn), nice, load weight log, burst total/restant/exécuté (log-ms), wait log-ms, response log-ms, turnaround log-ms, vruntime delta signé log-ms, deadline slack signé log-ms, ctx_switches log, io_ratio, has_deadline, remaining_ratio, observed. Toutes les durées sont en log-space pour compresser les distributions à longue queue.

**Features globales (13 dims) :** clock log-ms, history fill ratio, total tasks log, ratios observed/completed/ready/sleeping/rt_runnable/cfs_runnable, cpu_busy_ratio, has_current, current_is_rt, current_runtime log-ms.

### Pipeline forward

```
candidate_features → task_encoder (34→128→128) ─────────────────────────────────────┐
history_features   → task_encoder → +recency_bias → history_adapter → cross-attn ───┤
all_task_features  → task_encoder → self-attn (set refinement) → cross-attn ────────┤→ gate (3) → scorer (256→128→1) → logit par candidat
global_features    → global_encoder (13→128) ────────────────────────────────────────┘
```

- **task_encoder** : MLP 2 couches avec ReLU et Dropout (34→128→128)
- **history_adapter** : MLP 2 couches (128→128→128) + biais de récence relatif linéaire appris
- **set refinement** : `num_layers` tours de self-attention sur `all_task_features` (permet au modèle de comparer les tâches entre elles)
- **cross-attention** : chaque candidat attende l'historique et l'ensemble global (dot-product attention, dim 128)
- **gate** : projection 4×128→3 + sigmoid, pondère dynamiquement les 3 sources de contexte (historique, all_tasks, global) par candidat
- **scorer** : MLP 256→128→1 avec ReLU et Dropout, produit un logit scalaire par candidat
- **sortie** : logits masqués (−∞ sur les non-candidats), le `argmax` est la décision

**Taille** : ~150k paramètres. Entraîné sur CPU en ~5 min pour 20 epochs sur 28k décisions.

### Hyperparamètres par défaut

| Paramètre | Valeur |
|-----------|--------|
| `task_hidden_dim` | 128 |
| `history_hidden_dim` | 128 |
| `num_layers` | 1 |
| `dropout` | 0.1 |
| `lr` | 3e-4 (AdamW) |
| `weight_decay` | 1e-4 |
| `batch_size` | 256 |
| `max_history` | 256 |

---

## Teacher et signal d'apprentissage

### Problème avec l'imitation directe de CFS

Avec `--teacher cfs`, le modèle apprend à copier CFS : `loss = cross_entropy(logits, cfs_index)`. Deux limites :

1. **CFS n'est pas optimal** : il optimise l'équité (vruntime), pas le turnaround ni le response time.
2. **Signal pauvre** : un seul index gagnant — tous les autres candidats sont traités identiquement comme "mauvais", même si certains sont quasi aussi bons.

### Teacher heuristique (`--teacher heuristic`)

Remplace le label unique par un **score continu par candidat** inspiré de HRRN, CFS et des politiques à deadline :

```
u_i = 1.5 × wait / (remaining + 10)         # ratio attente/burst (HRRN)
    + 2.0 × is_new × wait / (remaining + 20) # aging des nouvelles tâches
    + deadline_bonus                          # urgence si slack faible
    - 0.004 × remaining                      # pénalise les longs bursts (SRPT-like)
    - 0.25 × max(vruntime_delta, 0)           # pénalise le déséquilibre d'équité CFS
    - switch_cost                             # malus si préemption sans gain net
```

Avec des règles spéciales : la tâche courante avec peu de burst restant est maintenue (évite les préemptions inutiles) ; une nouvelle tâche avec un response time élevé est promue.

### Loss combinée (3 termes)

| Terme | Poids | Rôle |
|-------|-------|------|
| Cross-entropy `CE(logits, argmax(scores))` | 0.2 | Ancre sur le meilleur candidat |
| KL-divergence listwise `KL(log_softmax(logits) ∥ softmax(scores/τ))` | 1.0 | Apprend le ranking complet, pas juste le gagnant |
| Pairwise BCE sur `logit_i − logit_cfs` vs `sign(score_i − score_cfs)` | 0.2 | Force le modèle à distinguer ce qui bat CFS |

Température teacher **τ = 0.7** (durcit légèrement la distribution, évite un signal trop uniforme).

---

## Données d'entraînement

- **50 workloads** synthétiques × **8 seeds** × **260 ms** de simulation = 400 épisodes, ~28k décisions
- Collecte sous politique **CFS** (les états sont ceux que CFS génère)
- Split **70/15/15** au niveau épisode (pas décision) pour éviter la fuite de données inter-épisodes
- Chaque sample encode un instant de scheduling : features des candidats, historique, all_tasks, global, label teacher

---

## Évaluation

### Offline (imitation)

Top-k accuracy sur le split test : le modèle prédit-il le même candidat que le teacher ?

```bash
cd model
uv run python scripts/evaluate_baseline.py \
    --dataset-dir artifacts/candidate_dataset_large_heuristic \
    --checkpoint artifacts/training_runs/candidate_large_heuristic/best_model.pt \
    --artifacts-json artifacts/training_runs/candidate_large_heuristic/dataset_artifacts.json \
    --run-config artifacts/training_runs/candidate_large_heuristic/run_config.json \
    --ce-weight 0.2 --listwise-weight 1.0 --pairwise-weight 0.2 --teacher-temperature 0.7
```

Résultat obtenu (teacher heuristique + loss listwise) : **top-1 = 91.6%, top-3 = 99.9%**.

### Online (closed-loop, métriques réelles)

Le modèle prend les vraies décisions dans l'émulateur, on compare les métriques finales contre CFS pur sur les mêmes workloads et seeds :

```bash
cd model
uv run python scripts/compare_policies.py \
    --taskfiles ../emulator/tasks/demo.tasks ../emulator/tasks/generated/workload_000.tasks \
    --model-run-dir artifacts/training_runs/candidate_large_heuristic \
    --seeds 0 1 2 3 4 \
    --duration 260 \
    --output artifacts/evaluation/policy_comparison.json
```

Résultats sur 60 runs (12 workloads × 5 seeds), positif = amélioration vs CFS :

| Métrique | Heuristique | Modèle |
|----------|-------------|--------|
| avg_response_ms | +12.7% | **+10.5%** |
| avg_wait_ms | +1.0% | **+1.4%** |
| avg_turnaround_ms | +0.7% | **+0.9%** |
| ctx_switches | +11.4% | +0.8% |
| fairness_index | −0.6% | −0.6% |

Le modèle bat CFS sur les métriques principales. Le gap sur `ctx_switches` vs l'heuristique est attribuable au covariate shift (voir ci-dessous).

---

## Threats to validity

**Données**

- Les états collectés sont ceux de CFS, pas ceux que le modèle génère en closed-loop → **covariate shift**. En production, le modèle rencontre des états jamais vus à l'entraînement. DAgger (Ross et al.) y remédierait : jouer le modèle en closed-loop, relabelliser les états induits avec le teacher, réentraîner.
- 50 workloads synthétiques, la distribution peut ne pas couvrir les charges réelles (pas de tâches I/O-bound réalistes, pas de burst multi-phases).

**Simulateur**

- L'émulateur n'est pas le vrai scheduler Linux : pas de préemption timer hardware, pas de NUMA, pas d'interruptions, pas de groupes de cgroups.
- Les simulations sont courtes (260 ms) ; le comportement à long terme ou sous charge soutenue n'est pas évalué.

**Modèle**

- La précision d'imitation à 91.6% ne garantit pas la qualité online : les 8.4% d'erreurs peuvent tomber sur des décisions critiques (ex. éviter la famine, gérer une deadline).
- Le gain mesuré (+10.5% response time) dépend de la qualité du teacher heuristique — si ses poids sont mal calibrés, le modèle apprend ses défauts.
- Pas d'évaluation de robustesse sous distribution shift (workloads hors distribution).

## Structure

```text
scripts/
    build_candidate_dataset.py   génère le dataset depuis l'émulateur
    dataset.py                   dataset PyTorch + collate
    model.py                     CandidateSchedulerModel
    common.py                    entraînement / évaluation (loss listwise + pairwise)
    train_baseline.py            entraînement standard
    train_distillation.py        entraînement d'un student par distillation
    evaluate_baseline.py         évaluation offline (top-k accuracy)
    evaluate_distilled.py        évaluation du modèle distillé
    compare_policies.py          comparaison online CFS / heuristique / modèle
    sweep_train.py               sweep WandB
artifacts/
    candidate_dataset/           dataset généré par l'émulateur
    training_runs/               checkpoints et configs
    evaluation/                  rapports d'évaluation
```

## Dépendances

Depuis la racine du repo:

```bash
uv sync --project model
```

## 1. Construire le dataset

Le dataset est généré directement à partir de l'émulateur. Trois modes de supervision sont disponibles:

- `heuristic`: **recommandé** — score multi-critères par candidat (HRRN + aging + deadline + équité), loss listwise + pairwise;
- `cfs`: rapide, imite les décisions de `CFS` (plafonne à la qualité de CFS);
- `oracle`: plus lent, étiquette chaque état avec le meilleur candidat contrefactuel trouvé par l'émulateur.

Pour enrichir fortement les workloads avant la génération:

```bash
python3 emulator/generate_workloads.py \
    --output-dir emulator/tasks/generated \
    --count 48 \
    --seed 42
```

Exemple rapide:

```bash
cd model
uv run python scripts/build_candidate_dataset.py \
    --taskfiles ../emulator/tasks/demo.tasks ../emulator/tasks/all_in_one.tasks \
    --output-dir artifacts/candidate_dataset \
    --policies CFS \
    --seeds 0 1 2 3 4 5 \
    --duration 220 \
    --teacher cfs \
    --max-history 256
```

Exemple plus ambitieux:

```bash
cd model
uv run python scripts/build_candidate_dataset.py \
    --taskfiles ../emulator/tasks/demo.tasks ../emulator/tasks/all_in_one.tasks \
    --output-dir artifacts/candidate_dataset_oracle \
    --policies CFS \
    --seeds 0 1 2 \
    --duration 220 \
    --teacher oracle \
    --max-history 256
```

Le mode `oracle` est volontairement plus coûteux, parce qu'il rejoue plusieurs futurs possibles par décision.

## 2. Entraîner le modèle

```bash
cd model
uv run python scripts/train_baseline.py \
    --dataset-dir artifacts/candidate_dataset_large_heuristic \
    --output-dir artifacts/training_runs/candidate_large_heuristic \
    --batch-size 256 \
    --epochs 20 \
    --lr 3e-4 \
    --max-history 256 \
    --num-workers 4 \
    --ce-weight 0.2 \
    --listwise-weight 1.0 \
    --pairwise-weight 0.2 \
    --teacher-temperature 0.7
```

Options utiles:

- `--task-hidden-dim` / `--history-hidden-dim` / `--num-layers` / `--dropout`
- `--ce-weight` / `--listwise-weight` / `--pairwise-weight` — poids des 3 termes de loss
- `--teacher-temperature` — température τ pour le term listwise (défaut 0.7)
- `--init-checkpoint` — fine-tuner un checkpoint existant
- `--wandb-mode offline|online`

Le dossier de run contient:

- `best_model.pt`
- `dataset_artifacts.json`
- `run_config.json`
- `training_history.json`
- `final_results.json`

## 2b. Finetune oracle après pré-entraînement CFS

Le curriculum recommandé est:

1. entraîner sur `teacher=cfs`;
2. fine-tuner le checkpoint obtenu sur `teacher=oracle`.

```bash
cd model
uv run python scripts/finetune_oracle.py \
    --dataset-dir artifacts/candidate_dataset_oracle \
    --init-checkpoint artifacts/training_runs/candidate_baseline/best_model.pt \
    --output-dir artifacts/training_runs/candidate_oracle_finetune \
    --batch-size 256 \
    --epochs 10 \
    --lr 1e-4
```

## 3. Évaluer le modèle

```bash
cd model
uv run python scripts/evaluate_baseline.py \
    --dataset-dir artifacts/candidate_dataset \
    --checkpoint artifacts/training_runs/candidate_baseline/best_model.pt \
    --artifacts-json artifacts/training_runs/candidate_baseline/dataset_artifacts.json \
    --run-config artifacts/training_runs/candidate_baseline/run_config.json \
    --output-dir artifacts/evaluation/candidate_baseline \
    --batch-size 256 \
    --num-workers 4
```

Sorties principales:

- `eval_metrics.json`
- `selection_breakdown.json`

## 4. Distillation

La distillation entraîne un **student plus petit** à partir du meilleur checkpoint du baseline.

```bash
uv run python scripts/train_distillation.py \
    --parsed-dir artifacts/candidate_dataset \
    --teacher-checkpoint artifacts/training_runs/cadidate_baseline/best_lstm_model.pt \
    --output-dir artifacts/training_runs/lstm_distilled \
    --batch-size 512 \
    --epochs 5 \
    --lr 3e-3 \
    --window-size 64 \
    --num-workers 4 \
    --temperature 4.0 \
    --alpha 0.5 \
    --student-embed-dim 64 \
    --student-hidden-size 128 \
    --student-num-layers 1 \
    --student-dropout 0.0 \
    --weight-decay 1e-5 \
    --early-stopping-patience 3
```

Produit dans `artifacts/training_runs/lstm_distilled/` :
- `best_student_model.pt`
- `training_history.json`
- `run_config.json`
- `dataset_artifacts.json`
- `final_results.json`

## 5. Évaluation du modèle distillé

```bash
uv run python scripts/evaluate_distilled.py \
    --parsed-dir artifacts/parsed_sched_switch \
    --checkpoint artifacts/training_runs/lstm_distilled/best_student_model.pt \
    --artifacts-json artifacts/training_runs/lstm_distilled/dataset_artifacts.json \
    --run-config artifacts/training_runs/lstm_distilled/run_config.json \
    --output-dir artifacts/evaluation/lstm_distilled \
    --batch-size 1024 \
    --num-workers 4 \
    --top-n-confusion 15
  ```

Produit dans `artifacts/evaluation/lstm_distilled/` :
- `eval_metrics.json`
- `per_class_accuracy.json`
- `classification_report.txt`
- `confusion_matrix_top15.png`
- `distillation_summary.json`

## 6. Rejouer le modèle dans l'émulateur

Le modèle entraîné peut ensuite être comparé à `CFS` directement dans l'émulateur.

Mode `shadow`:

```bash
uv run --project model python ../emulator/sched_em.py ../emulator/tasks/demo.tasks \
    -p CFS \
    -d 220 \
    --seed 3 \
    --model-mode shadow \
    --model-run-dir artifacts/training_runs/candidate_baseline \
    --stats-json ../emulator/artifacts/benchmarks/demo_shadow.json
```

Mode `closed-loop`:

```bash
uv run --project model python ../emulator/sched_em.py ../emulator/tasks/demo.tasks \
    -p CFS \
    -d 220 \
    --seed 3 \
    --model-mode closed-loop \
    --model-run-dir artifacts/training_runs/candidate_baseline \
    --stats-json ../emulator/artifacts/benchmarks/demo_closed_loop.json
```

Les comparaisons portent sur:

- `avg_wait_ms`
- `avg_turnaround_ms`
- `avg_response_ms`
- `fairness_index`
- `ctx_switches`
- `starvation_count`

## SLURM

Une fois le dataset construit, les soumissions utilisent le même pipeline:

```bash
sbatch submit.sh
sbatch submit_eval.sh
sbatch submit_sweep_train.sh
sbatch submit_distill.sh
sbatch submit_distill_eval.sh
```
