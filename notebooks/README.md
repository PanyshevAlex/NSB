# notebooks/

Рабочие Jupyter-ноутбуки исследования. Один ноутбук ≈ одна фаза из [`../framework.md`](../framework.md).

## Конвенции

- **Драйв через Jupyter MCP**, не ручная правка JSON. Kernel — conda-env **`tgb`**
  (см. раздел «Jupyter notebooks» в `../CLAUDE.md`).
- **Визуализация — Plotly**, датафреймы — **Polars**, текст/комментарии — **на русском**.
- Подключение к любому ноутбуку: `use_notebook(notebook_path="notebooks/<file>.ipynb")`
  (`DOCUMENT_ID` в `.mcp.json` — лишь дефолтный landing-док, не ограничение).

## Naming

`p{phase}_{slug}.ipynb` — префикс фазы держит ноутбуки в порядке исследования и даёт
соответствие 1:1 фазам framework.

| ноутбук | фаза | назначение |
|---|---|---|
| `p0_baselines.ipynb` | P0 | воспроизведение и якорь baseline'ов: наши числа vs Table 1 (через MLflow) |
| `p1_eda_user_item.ipynb` | P1 | EDA в разрезе user–item на `tgbn-genre` (жива ли идея?) |
| `p2_error_analysis.ipynb` | P2 | error analysis TGNv2 в user–item разрезе |

Ноутбуки последующих фаз (P5+) добавляем по мере перехода к ним.

> `experiments/analysis.ipynb` — старый scratch-ноутбук, заменён на `p0_baselines.ipynb`.
