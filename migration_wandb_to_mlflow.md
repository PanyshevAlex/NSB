# Миграция трекинга экспериментов: Weights & Biases → локальный MLflow

> Что сделал Claude Code в рамках ресерча: заменил облачный **wandb** на локальный **MLflow**
> (SQLite-бэкенд, без аккаунта и сети). Затронуто **3 файла**. Ниже — что и зачем изменилось,
> с before → after.
>
> *(Документ описывает только миграцию wandb→mlflow. Строки про `--dump_eval_preds` в текущем
> diff — это отдельная более поздняя фича анализа ошибок, не часть миграции.)*

## Зачем (мотивация)

| Было (wandb) | Стало (MLflow) |
|---|---|
| Облако, нужен аккаунт/`entity`, сеть | **Локально**: SQLite + папка `./mlruns`, офлайн |
| Гиперпараметры в `config` | **Params + tags = источник правды** (фильтрация/группировка) |
| Имя рана кодировало все гиперпараметры | Имя — короткий ярлык; полный slug → в тег |
| `wandb.log` без явного `step` (дробная «эпоха» как метрика) | `mlflow.log_metrics(..., step=<int>)` — корректная ось X |
| Падение рана оставляло «висящий» статус | `with mlflow.start_run()` — крах помечается `FAILED` |
| — | Провенанс: git-commit/dirty, hostname, версии torch/python |

## Затронутые файлы

| Файл | Изменение |
|---|---|
| `requirements.txt` | `wandb==0.17.3` → `mlflow==3.11.1` |
| `utils/args.py` | + флаг `--experiment` (имя research-треда в MLflow) |
| `train-tgbn-nodeproppred.py` | импорт, инициализация, все вызовы логирования, params/tags/provenance/summary/artifacts |

---

## 1. Зависимость

```diff
- wandb==0.17.3
+ mlflow==3.11.1
```

## 2. Импорты

```diff
- import wandb
+ import mlflow
+ import platform, socket, subprocess   # для provenance-тегов
```

## 3. Инициализация рана (ядро миграции)

**Было:**
```python
wandb.init(
    project='tgnv2',
    entity='rossignol',     # привязка к облачному аккаунту
    name=run_name,          # имя = длинный slug со всеми гиперпараметрами
    config=args,
)
```

**Стало:**
```python
# Локальный трекинг: SQLite backend store + ./mlruns artifact store
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment(args.experiment)              # research-тред (baselines, user-item-matrix)
short_name = '{}-{}-s{}'.format(model_name, name, seed)  # короткий ярлык; slug → в тег

with mlflow.start_run(run_name=short_name):         # crash-safe: крах → статус FAILED
    mlflow.log_params(vars(args))                   # полный конфиг — в params
    mlflow.set_tags({                               # то, по чему фильтруем, но не тюним
        "model": model_name, "dataset": name, "seed": seed,
        "slug": run_name, "device": str(device),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "git_commit": git_sha, "git_dirty": git_dirty,   # провенанс
    })
    ...  # обучение
```

## 4. Логирование метрик: `wandb.log` → `mlflow.log_metrics(step=int)`

**Было** (стриминговая кривая обучения, дробная «эпоха» как метрика, без явного step):
```python
metrics = {
    "train/loss": total_loss / num_label_ts,
    "train/epoch": count / train_loader_length + epoch,   # дробное значение, не ось
    f"train/{eval_metric}": total_score / num_label_ts,
}
wandb.log(metrics)
```

**Стало** (монотонный целочисленный `step`, сквозной по всем эпохам):
```python
# одна точка на label-timestamp, индекс — глобальный шаг через все эпохи
mlflow.log_metrics(
    {"train/loss": total_loss / num_label_ts,
     f"train/{eval_metric}": total_score / num_label_ts},
    step=global_step,
)
global_step += 1
```

Эпоховые сводки — namespace `epoch/`, индекс = номер эпохи:
```diff
- "train/train_loss": ...,  "train/lr": lr_scheduler.get_lr(),
- wandb.log(metrics)
+ "epoch/train_loss": ...,  "epoch/lr": float(lr_scheduler.get_lr()[0]),
+ mlflow.log_metrics(metrics, step=epoch)
```

В `test()` — то же: `wandb.log(metric_dict)` → `mlflow.log_metrics(metric_dict, step=epoch)`
(функции добавлен параметр `epoch`).

## 5. Новое: provenance, summary-метрики, артефакты

**Хелпер провенанса** (новый):
```python
def _git_commit_and_dirty():
    """Git commit SHA + dirty-флаг — в теги MLflow для воспроизводимости."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
    return sha, dirty
```

**Итоговые числа рана** (заголовочные метрики для статьи) и **артефакт лога** — в конце рана:
```python
mlflow.log_metrics({
    "best_val_ndcg": max_val_score,
    "best_test_ndcg": max_test_score,
    "best_epoch": best_test_idx + 1,
})
mlflow.log_artifact(log_full_path)    # текстовый лог как артефакт рана
```

## 6. Просмотр результатов

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## Итог (для слайда)

- **3 файла**, один импорт, один блок инициализации, ~4 вызова логирования — и трекинг стал
  **локальным, офлайн, воспроизводимым**.
- Ключевые проектные решения, заложенные при миграции: *имя рана = короткий ярлык; params+tags =
  источник правды; целочисленный `step`; SQLite-бэкенд; git-провенанс в тегах; crash-safe ран.*
- Эти конвенции зафиксированы в `CLAUDE.md` (раздел «Experiment tracking»), чтобы каждый
  следующий ран им следовал автоматически.
