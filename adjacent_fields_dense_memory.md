# Смежные области с идеей «плотная per-pair матрица-память» и как адаптировать

> Сводка deep-research (5 углов вне temporal graphs → 21 источник → 96 claim'ов → 25
> верифицировано adversarial-голосованием → **24 подтверждено / 1 убито** → 8 тем).
> Цель: найти аналоги нашей идеи (плотное `N_user × N_item` состояние, дешёвая запись с каждого
> события, чтение для предсказания) в смежных доменах и придумать адаптацию под TGB node affinity.
> Связано с `framework.md`, `backlog.md`, `lit_review_cold_start_user_item.md`.

## TL;DR

Наша идея — **устоявшийся single-model механизм** (не ансамбль) сразу в нескольких полях:

1. **Linear attention / SSM** (GLA, RetNet, Mamba2, DeltaNet, Gated DeltaNet): скрытое состояние —
   это **матрица** = затухающая сумма внешних произведений key-value = плотная per-pair
   associative memory. Забывание/устойчивость → **обучаемое decay-gating**.
2. **Fast weights / FWP / modern Hopfield**: формально **эквивалентны** linear attention; Hebbian
   запись `A(t)=λA(t−1)+η·hhᵀ` — мгновенная запись с *одного* примера (= наш cold-start instant-write).
3. **Внешняя память** (NTM, DNC, Key-Value MemNN): явная плотная `N×M` матрица, запись —
   **gated erase-then-add** (= обучаемый per-cell decay/EMA).
4. **RUM (recsys, WSDM 2018)** — **самый прямой аналог**: персональная матрица памяти на юзера +
   **общая на всех Global Latent Feature Table** = «тёплые общие item/feature-факторы + персональное
   состояние», ровно наш рычаг асимметрии `items ≪ users`.

**Синтез-рецепт для нас:** держать плотное `N_user × N_item` с **обучаемым per-(user,item)
затуханием** (= EMA / диагональный линейный SSM), писать каждое событие user→item как
**delta / erase-then-add** (а не чистым накоплением — иначе насыщение), и **подпирать холодные
строки общими тёплыми item-факторами** (RUM-style). Одна модель, чтение = один matvec, дёшево на CPU.

> ⚠️ **Главный caveat (entity-space vs feature-space):** в linear-attention/fast-weights матрица
> живёт в **feature-пространстве** (`d_key × d_value`), а НЕ буквально `N_user × N_item`.
> Доказанная в литературе эквивалентность — на уровне **рекуррентности/механизма** (затухающая
> сумма outer-product = EMA = диагональный SSM), не index-for-index. **Переиндексация в
> entity-пространство — наш собственный, непроверенный вклад**: механизмы (decay-gating, delta-rule,
> erase-then-add) переносятся, но благоприятные гарантии (associative recall, capacity bounds)
> доказаны для фич, не для сущностей.

## Таблица: аналог → механизм cold-start/устойчивости → адаптация к TGB

| Аналог | Форма плотной памяти | Механизм cold-start / forgetting / устойчивости записи | Адаптация к `N_user × N_item` affinity |
|---|---|---|---|
| Linear attention / GLA / RetNet / Mamba2 | `S = Σ kₜvₜᵀ` (сумма KV outer-product) | обучаемое decay-gating: скаляр (RetNet), data-dependent скаляр (Mamba2), **per-row/per-key (GLA)** | per-(user,item) обучаемое затухание `λ_{u,i}` на плотном состоянии = EMA / диагональный SSM; чтение `S` ключом-юзером = matvec |
| Delta rule / DeltaNet / Gated DeltaNet | та же матрица, **error-correcting** запись | прочитать старое значение ключа, записать **остаток (delta)** + gating для быстрого стирания — лечит насыщение/интерференцию чистого накопления | на свежем событии user→item **перезаписывать** ассоциацию `(u,i)` (delta/erase-then-add), а не слепо накапливать; gate чистит устаревшие строки |
| Fast Weights / FWP (Ba/Hinton; Schlag) | `A(t)=λA(t−1)+η·outer` ; `A=Σ vₖ⊗kₖ` | **мгновенная one-shot Hebbian запись** с одного примера; `λ` управляет забыванием; чтение = attention-with-decay | первая же интеракция пишет строку сразу (cold-start instant-write); чтение = decayed attention |
| Modern Hopfield nets | матрица хранимых паттернов; softmax-чтение | retrieval за **один шаг**; правило = transformer attention | принципиальное «читать плотную матрицу для предсказания» как attention над item-паттернами (но softmax, не linear-time) |
| Meta-learning Hebbian fast weights | slow-веса (SGD) + fast Hebbian-матрица | fast-веса связывают метку с репрезентацией из **одного** примера (мета-обученное правило записи) | медленные GNN-параметры + **быстрая** per-(user,item) запись, мета-обученная быть cold-start-устойчивой |
| **RUM** (User Memory Networks, recsys) | матрица на юзера `M_u`; **общая на всех** Global Latent Feature Table `F` | немного **общих тёплых** латентных фич адресуют персональную память; item attends к общим фичам `w_{ik}=q_iᵀf_k` | эксплуатировать `items ≪ users`: плотное per-user состояние + **подпорка холодных строк немногими общими тёплыми item/feature-факторами** |
| NTM / DNC / Key-Value MemNN | внешняя плотная `N×M` матрица `M_t` | **gated erase-then-add**: `M←M(1−w·e)+w·a`, `e∈(0,1)ᴹ`; KV-MemNN — **асимметричные** key(адрес)/value(выход) кодировки | обучаемый per-cell erase/add = per-(user,item) decay/EMA онлайн-запись; раздельные кодировки «адресация (user)» vs «выход (affinity)» |

---

## Находки с цитатами

### A. Матричное состояние = плотная per-pair память; чтение = attention (3-0)
Linear-attention/SSM держат matrix-valued hidden state — затухающую сумму KV outer-product — и это
**формально эквивалентно** Fast Weight Programmers (`W_t = Σ vτ⊗kτ`).
Источники: [GLA 2312.06635](https://arxiv.org/abs/2312.06635) (ICML 2024),
[Linear Transformers=FWP 2102.11174](https://arxiv.org/abs/2102.11174) (ICML 2021),
[Fast Weights 1610.06258](https://arxiv.org/abs/1610.06258) (NIPS 2016),
[FWP primer 2508.08435](https://arxiv.org/abs/2508.08435).
**→ Для нас:** чтение нашей `M[u,:]` — принципиальная linear-attention операция, а не хак.

### B. Обучаемое затухание = EMA / диагональный SSM, разной гранулярности (3-0)
RetNet — скаляр; Mamba2 — data-dependent скаляр; **GLA — per-row (per-key) data-dependent decay**.
Источники: [GLA 2312.06635](https://arxiv.org/abs/2312.06635), [FWP primer 2508.08435](https://arxiv.org/abs/2508.08435).
**→ Для нас:** наш обучаемый `λ_{u,i}` обоснован; **GLA per-row decay = per-user-строка decay** — самый близкий аналог (backlog **H5/T7**).

### C. ⭐ Чистое накопление насыщается → нужен delta-rule + gating (3-0)
Чисто аддитивное outer-product накопление имеет предел ёмкости (насыщение/интерференция); фикс —
**delta rule** (прочитать старое значение ключа, записать остаток) + **gating** для быстрого стирания.
Источники: [Schlag 2102.11174](https://arxiv.org/abs/2102.11174),
[DeltaNet 2406.06484](https://arxiv.org/abs/2406.06484) (NeurIPS 2024),
[Gated DeltaNet 2412.06464](https://arxiv.org/abs/2412.06464) (ICLR 2025), [FWP primer 2508.08435](https://arxiv.org/abs/2508.08435).
**→ Для нас (важнейший практический инсайт):** правило обновления `M` должно быть **delta/erase-then-add**,
а не просто EMA-накопление — иначе тёплые юзеры «забивают» матрицу. Прямо уточняет backlog **H5/T7**.

### D. Cold-start instant-write: запись строки с первого примера (3-0)
Fast weights пишут плотную ассоциативную матрицу Hebbian-правилом `A(t)=λA(t−1)+η·hhᵀ` — мгновенно,
с одного примера. Modern Hopfield — retrieval за один шаг (= transformer attention). Meta-learning
fast weights — slow+fast веса связывают метку из одного примера.
Источники: [1610.06258](https://arxiv.org/abs/1610.06258),
[Hopfield 2008.02217](https://arxiv.org/abs/2008.02217) (ICLR 2021),
[Meta Hebbian 1807.05076](https://arxiv.org/abs/1807.05076).
**→ Для нас:** обоснование, что плотная память чинит cold-start (пишем строку с 1-го взаимодействия) — backlog **H2**.

### E. ⭐ RUM: тёплые общие факторы + персональная память (3-0)
RUM держит персональную `M_u`, но **Global Latent Feature Table `F` общая на всех юзеров**; item
адресует общие фичи `w_{ik}=q_iᵀf_k`.
Источник: [RUM, WSDM 2018](https://dl.acm.org/doi/10.1145/3159652.3159668).
**→ Для нас:** самый прямой рецепт item-side достройки холодных строк (backlog **H3/T5**) — держать
плотное per-user состояние, но **подпирать его немногими общими тёплыми item-факторами**.

### F. Внешняя память: gated erase-then-add + асимметричные key/value (3-0)
NTM/DNC пишут `M←M(1−w·e)+w·a` (вдохновлено input/forget-гейтами LSTM; при `e=1` → EMA с обучаемой
скоростью). KV-MemNN: ключ под адресацию, значение под выход — **асимметричные кодировки**.
Источники: [NTM 1410.5401](https://arxiv.org/abs/1410.5401),
[DNC Nature 2016](https://www.nature.com/articles/nature20101),
[KV-MemNN 1606.03126](https://arxiv.org/abs/1606.03126).
**→ Для нас:** per-cell erase/add = наш per-(user,item) decay; и идея **раздельных кодировок**
адресации (user) vs выхода (affinity).

---

## Синтез: конкретный дизайн обновления `M` (черновик для T2/T7)

```
# состояние: M ∈ R^{N_user × N_item} (scalar/cell), плюс общие тёплые item-факторы V (RUM-style)
# на событие (u, i, t):
#   1) decay (GLA per-row):     M[u, :] ← λ_u ⊙ M[u, :]        # λ_u обучаемый, per-user-row
#   2) delta/erase-then-add:    old = M[u, i]
#                               M[u, i] ← old + β·(write(msg_{u,i,t}) − old)   # delta, не слепое +=
#   (β, λ_u — обучаемые; write — малый learnable φ)
# чтение для предсказания юзера u:
#   logits_u = head( z_u_GNN , M[u, :] , V )   # ОДНА голова: GNN-эмбеддинг + строка памяти + тёплые факторы
#   холодная строка M[u,:] разрежена → V (тёплые item-факторы) её "достраивают"
```
Гранулярность затухания: начать с per-user-row `λ_u` (GLA-аналог), не per-cell (дёшево); правило
записи — delta (находка C), не чистый EMA. Холодные строки — через общие `V` (находка E).

## Honest caveats (из верификации)

1. **Entity-space vs feature-space** (главный): теория доказана для `d_key×d_value`, не
   `N_user×N_item`. Переиндексация — наш непроверенный вклад; переносим механизмы, не гарантии.
2. **«Cold-start»** — наша доменная аналогия везде, кроме RUM (и тот про *sequential* rec, не явно cold-start).
3. **Категории (4) online/streaming MF + inductive matrix completion и (5) Complementary Learning
   Systems дали 0 подтверждённых claim'ов** в этом проходе — считать неподтверждёнными здесь
   (частично закрыты только RUM и meta-Hebbian). Стоит отдельный прицельный поиск, если нужны эти опоры.
4. **Hopfield-чтение — softmax** (высокая ёмкость, но не linear-time) → не наследует дешёвый CPU-matvec.
5. **Refuted (1-2):** claim, что DNC content-read = взвешенная сумма по строкам = аналог нашей
   `N_rows`-асимметрии — **не прошёл**; выжил только erase-then-add **write** DNC.

## Открытые вопросы = решения по дизайну метода

1. Сохраняет ли переиндексация память из feature- в entity-пространство благоприятные свойства
   (associative recall, delta-rule, capacity, EMA-устойчивость)? В источниках entity-indexed память не тестировалась.
2. Лучшая схема подпорки холодных строк тёплыми item-факторами: low-rank `S≈U·Vᵀ` (тёплый `V`) vs
   регуляризация строки к общему prior vs буквальная плотная строка (RUM = общие фичи)?
3. Есть ли принципиальный online-MF decay/regularization-to-prior для устаревших строк сверх нашего
   linear-SSM EMA? (категория 4 не дала verified-источников — нужен отдельный поиск).
4. **Guardrail:** slow GNN-параметры + fast per-(user,item) запись, читаемые в одном forward pass —
   это **одна модель** или скрытый ансамбль? Должно быть единой головой (находка F: раздельные
   кодировки, но одно предсказание), иначе нарушает no-ensemble.

## Следствия для backlog

- **H5/T7 (правило обновления):** заменить «EMA vs MA-k vs last» на **delta/erase-then-add (DeltaNet)
  vs чистый EMA**; per-row `λ_u` (GLA) как стартовая гранулярность затухания. ⭐ поднять приоритет.
- **H3/T5 (item-side достройка холодных):** взять **RUM-схему** (общие тёплые факторы + персональное
  состояние) как конкретную реализацию.
- **H4 (M — активный ингредиент):** чтение `M` как linear-attention/Hopfield-операция — обосновано.
- **Новый риск-пункт:** entity-space переиндексация — наш непроверенный вклад; добавить sanity-эксперимент.
- **Позиционирование (related work):** наш метод = entity-indexed, per-(user,item) специализация
  associative-memory/fast-weights/SSM линии, с асимметрией `items≪users` как рычагом feasibility.

## Источники (первичные, peer-reviewed, если не отмечено)

| Работа | id / DOI | Венью | Угол |
|---|---|---|---|
| Gated Linear Attention (GLA) | [2312.06635](https://arxiv.org/abs/2312.06635) | ICML 2024 | matrix-state / decay |
| DeltaNet | [2406.06484](https://arxiv.org/abs/2406.06484) | NeurIPS 2024 | delta-rule write |
| Gated DeltaNet | [2412.06464](https://arxiv.org/abs/2412.06464) | ICLR 2025 | gating + delta |
| Linear Transformers = Fast Weight Programmers | [2102.11174](https://arxiv.org/abs/2102.11174) | ICML 2021 | эквивалентность, delta |
| Using Fast Weights to Attend to the Recent Past | [1610.06258](https://arxiv.org/abs/1610.06258) | NIPS 2016 | Hebbian outer-product |
| FWP primer / survey (Irie & Gershman) | [2508.08435](https://arxiv.org/abs/2508.08435) | 2025 | обзор decay-гранулярности |
| Hopfield Networks is All You Need | [2008.02217](https://arxiv.org/abs/2008.02217) | ICLR 2021 | one-shot retrieval = attention |
| Metalearning with Hebbian Fast Weights | [1807.05076](https://arxiv.org/abs/1807.05076) | 2018 | slow+fast, one-shot |
| RUM: Sequential Rec. with User Memory Networks | [10.1145/3159652.3159668](https://dl.acm.org/doi/10.1145/3159652.3159668) | WSDM 2018 | ⭐ тёплые факторы + per-user |
| Neural Turing Machine | [1410.5401](https://arxiv.org/abs/1410.5401) | 2014 | erase-then-add write |
| Differentiable Neural Computer | [Nature 2016](https://www.nature.com/articles/nature20101) | Nature 2016 | gated memory write |
| Key-Value Memory Networks | [1606.03126](https://arxiv.org/abs/1606.03126) | EMNLP 2016 | асимметричные key/value |

*Метод: deep-research harness (fan-out → fetch → 3-голосная adversarial-верификация, kill при ≥2/3
refute → синтез). Категории online-MF и CLS не дали подтверждённых claim'ов — нужен отдельный проход.
Формулировки «cold-start» — доменные аналогии автора, кроме RUM.*
