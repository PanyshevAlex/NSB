# Обзор литературы: cold-start и user–item структура в temporal graph networks

> Сводка по результатам deep-research (5 углов поиска → 21 источник → 94 claim'а → 25
> верифицировано adversarial-голосованием → **23 подтверждено / 2 убито** → 8 тем).
> Привязка к проекту (`framework.md`, `backlog.md`): идея «плотная `N_user × N_item` память
> чинит cold-start, заимствуя тёплую item-side структуру». Литература актуальна до ~конца 2025.

## TL;DR

1. **Проблема доказана и центральна.** Memory-based TGN **не способны** выразить persistent
   forecast / moving average по сообщениям, и на TGB node-affinity простые эвристики стабильно
   бьют TGN/DyRep — ровно то, что мы воспроизвели в P2 (TGNv2 < persistent в 95%).
2. **Два известных single-model фикса** (без ансамблей, как требует наш guardrail): **TGNv2**
   (source-target identification — у нас в коде) и **NAViS** (Virtual State через эквивалентность
   эвристик и SSM, Oct 2025). NAViS — ключевой конкурент/смежная работа.
3. **Прямое эмпирическое подтверждение нашей гипотезы:** PTGCN показывает, что моделирование
   тёплой item-side динамики даёт **наибольший прирост именно холодным юзерам** (+40.8% / +28.5%
   NDCG@10 на двух самых разреженных группах MovieLens).
4. **Наш рычаг, похоже, не исследован:** ни один из разобранных методов **не материализует
   плотную per-(user,item) память** — используют inductive attention (TGAT), мета-обучение
   (few-shot) или per-node state. Это потенциальная новизна.

---

## 1. Проблема: представимостное узкое место TGN + эвристики побеждают

**Finding (confidence: high, vote 3-0).** Никакая формулировка TGN не выражает persistent
forecasting или moving average по сообщениям; эмпирически persistent forecast / moving average
над ground-truth метками стабильно и значимо бьют TGN/DyRep на всех четырёх TGB node-датасетах.

- TGB Table 4 (test NDCG@10 эвристик): trade 0.855, genre 0.509, reddit 0.559, token 0.508.
- Подтверждено независимо в NAViS: его Moving Average бьёт TGNv2 на всех четырёх датасетах;
  доказано, что RNN/LSTM/GRU не выражают Persistent Forecast.

Источники: [TGNv2 2411.03596](https://arxiv.org/abs/2411.03596) (NeurIPS 2024),
[TGB 2307.01026](https://arxiv.org/abs/2307.01026) (NeurIPS 2023),
[NAViS 2510.06940](https://arxiv.org/abs/2510.06940) (Oct 2025, под ревью).

**→ Для нас:** прямо совпадает с P2 (наш TGNv2 проигрывает persistent в 95%, 0.54 vs 0.87 на trade).
Это мотивация №1 проекта, подтверждённая тремя первичными источниками.

## 2. Известные single-model фиксы (не ансамбли)

**Finding (high, 3-0).** Оба SOTA-фикса — это изменения **дизайна состояния**, а не ансамбли:

- **TGNv2** — добавляет source-target (node-index) identification в каждое сообщение события;
  это *необходимо* для представления persistent forecast, moving average и класса
  авторегрессии. У нас реализовано в `models/msgmodule.py:EncodeIndexModule`
  (`msg = [z_src, z_dst, raw_msg, src_enc, dst_enc, t_enc]`).
- **NAViS** — вводит «Virtual State», эксплуатируя эквивалентность эвристик и линейных
  state-space models (Thm: EMA/SMA/Persistent Forecast — частные случаи линейных SSM).

Источники: [2411.03596](https://arxiv.org/abs/2411.03596), [2510.06940](https://arxiv.org/abs/2510.06940).

**→ Для нас (важно для позиционирования):** наша плотная `M[u,i]` с обучаемым затуханием —
это **per-(user,item) EMA = per-pair SSM**. NAViS делает *глобальный/виртуальный* SSM-state; мы —
*плотный per-pair*, что возможно только благодаря асимметрии `N_item ≪ N_user`. Это и отличает
нас от NAViS, и связывает с его теорией. **Метод обязан сравниваться с NAViS и TGNv2.**

## 3. Задача и bipartite-структура (TGB)

**Finding (high, 3-0).** TGB формализует dynamic node affinity prediction как
рекомендательную задачу (прогноз вектора частот взаимодействий `y_t[u,:]` по кандидат-узлам на
окно `[t, t+k]`, метрика **NDCG@10**). Датасеты genre/reddit/token — настоящие bipartite
user–item сети (юзеры × жанры / сабреддиты / токены), item'ов сильно меньше юзеров.

Источник: [TGB 2307.01026](https://arxiv.org/abs/2307.01026).

**Caveat:** `tgbn-trade` — nation-to-nation, **не** user-item bipartite. Точные числа узлов из
литературы были **REFUTED 0-3** (их не было в источнике) — но мы измерили их сами (см. ниже).

**→ Для нас:** наш EDA (`notebooks/scratch.ipynb`) закрывает open-question об асимметрии прямыми
замерами: genre 992 user × 513 item (≈1.9×), reddit 11 068 × 698 (≈16×), token 60 745 × 1 001
(≈61×); trade симметричен (254×254). Плотность пар genre 26% → плотная матрица дёшева и осмысленна.

## 4. Bipartite temporal-модели и per-entity состояние

**Finding (high, 3-0).** Per-entity динамическое состояние и коллаборативные inductive bias уже
используются:
- **JODIE** — две связанные RNN обновляют эмбеддинг юзера и item'а при каждом взаимодействии
  (TGN — обобщение JODIE как per-node-memory частного случая).
- **TGSRec** — рекомендация как Continuous-Time Bipartite Graph (непересекающиеся множества
  user/item) + Temporal Collaborative Transformer (query = таргет, key/value = bipartite-соседи).
- Прямой **TGN-for-recsys** моделирует взаимодействия как continuous-time bipartite граф.

Источники: [JODIE 1908.01207](https://arxiv.org/abs/1908.01207) (KDD 2019),
[TGSRec 2108.06625](https://arxiv.org/abs/2108.06625) (CIKM 2021),
[TGN-recsys 2403.16066](https://arxiv.org/html/2403.16066v2).

**→ Для нас:** наша плотная per-pair память — обобщение per-entity state (JODIE/TGN): вместо двух
векторов (user, item) храним явную ячейку на пару. (Caveat: «новизна» 2403.16066 завышена —
prior art есть; cold-start он не трогает.)

## 5. Перенос CF / co-occurrence cold-start в темпоральные графы

**Finding (high, 3-0).** Коллаборативные / co-occurrence сигналы над bipartite-структурой —
переносимый inductive bias и механизм cold-start:
- **DyGFormer** — neighbor co-occurrence encoding: считает, как часто сосед встречается в историях
  *обоих* концов; больше общих соседей → выше вероятность взаимодействия (CF-подобный bias;
  сильнейший компонент в абляциях).
- **CCFCRec** — единая contrastive-модель, переносящая co-occurrence сигналы в content-модуль и
  «вспоминающая» их на инференсе для исправления эмбеддингов холодных item'ов.

Источники: [DyGFormer 2303.13047](https://arxiv.org/abs/2303.13047) (NeurIPS 2023),
[CCFCRec 2302.02151](https://arxiv.org/abs/2302.02151) (WWW 2023).

**→ Для нас:** обоснование item-side достройки холодных строк `M[u,:]` (backlog **H3/T5**):
co-occurrence/CF над bipartite — рабочий, проверенный сигнал. (Caveat: «CF» для DyGFormer —
аналогия ревьюера, не термин статьи.)

## 6. ⭐ Прямое подтверждение гипотезы: тёплая item-side структура чинит холодных

**Finding (high, 3-0).** **PTGCN** явно таргетит cold-start (юзеры с малой историей) и даёт
**наибольший прирост именно там**: моделирование high-order bipartite связности + item-side
темпоральной динамики подняло NDCG@10 на **+40.79%** и **+28.54%** над TiSASRec на двух самых
разреженных cold-start группах MovieLens; абляция, изолирующая item-динамику, даёт ~9–11%
(R/NDCG@5) и ~6–9% (R/NDCG@10) в среднем по трём датасетам.

Источник: [PTGCN 2107.05235](https://arxiv.org/abs/2107.05235) (ACM TOIS).

**→ Для нас:** **сильнейшее внешнее подтверждение** нашей логики (P2b: headroom от холодных +0.16…
+0.20). Caveat: PTGCN исключает *строго* нулевую историю (zero-interaction new users); «warm
item-side structure» — формулировка-аналогия, не дословная.

## 7. Существующие cold-start решения ≠ плотная per-pair память → новизна

**Finding (high, 3-0).** Cold-start/inductive методы для темпоральных графов **не** используют
плотное per-pair состояние:
- **TGAT** — inductive эмбеддинги новых/наблюдённых узлов через temporal graph attention
  (memoryless, без персистентного per-node state).
- **DLPNN** (шорткат по инициалам названия, не офиц. имя) — new-node link prediction как
  few-shot meta-learning.
- TGN-for-recsys cold-start вообще не трогает.

Источники: [TGAT 2002.07962](https://arxiv.org/abs/2002.07962) (ICLR 2020),
[2310.09787](https://arxiv.org/abs/2310.09787).

**→ Для нас:** **плотная `N_user × N_item` память выглядит неисследованным рычагом** — методы
используют attention/meta/per-node, но не плотную per-pair матрицу. Поддержка новизны (по аналогии
и мотивации, не прямым сравнением).

## 8. Параллельный бенчмарк/ограничение: TGB-Seq

**Finding (high, 3-0).** **TGB-Seq** (ICLR 2025) — бенчмарк на bipartite user-item recsys-данных
(ML-20M, Taobao, Yelp, GoogleLocal) — утверждает, что провал temporal-GNN архитектурный: ни
память, ни агрегация не различают item'ы, всегда co-occurring в одни и те же timestamps.

Источник: [TGB-Seq 2502.02975](https://arxiv.org/abs/2502.02975).

**Caveat:** режим `items ≪ users` тут **не универсален** — у Taobao item'ов БОЛЬШЕ юзеров
(863k vs 761k), Yelp лишь ~3.3×. Связь с moving-average границей TGN — аналогия ревьюера, не
утверждение статьи (механизм иной: идентичные timestamps + null-фичи).

---

## Что НЕ подтвердилось (убитые claim'ы — не опираться)

1. **Точные числа узлов TGB из литературы** (напр. «tgbn-genre 1505 узлов», «trade 255») —
   **REFUTED 0-3** (в источнике их нет). Качественная асимметрия `items ≪ users` для
   genre/reddit/token верна, но точные соотношения из литературы недоступны. → **Закрыто нами:**
   измерено прямо на диске в `notebooks/scratch.ipynb` (см. §3).
2. **«Temporal GNN проигрывают эвристике SGNN-HN на TGB-Seq»** — split-refuted (1-2). Не
   утверждать превосходство sequential-эвристики на TGB-Seq.

## Открытые вопросы (из отчёта) и наш статус

1. Точные `N_user / N_item` в genre/reddit/token? → **отвечено** нашим EDA (§3).
2. Материализует ли кто-то плотное per-(user,item) состояние? → по обзору **нет** (все — per-node /
   attention / meta). Это наш рычаг.
3. Как плотная per-pair память сравнится по NDCG@10 с moving-average, TGNv2, NAViS? → **никто не
   мерил** — это наш эксперимент (backlog T3/T9; добавить NAViS в baseline'ы).
4. Можно ли совместить тёплые item-факторы (CF cold-start) + source-target ID в **одной** модели
   без ансамбля, и какой компонент реально двигает cold-start? → открыто (backlog H3/H1).

## Следствия для backlog

- **Добавить NAViS ([2510.06940](https://arxiv.org/abs/2510.06940)) в baseline'ы** рядом с TGNv2 и
  эвристиками — это прямой конкурент с близкой (SSM) теорией.
- **Позиционирование метода:** плотная `M[u,i]` с обучаемым λ = per-(user,item) EMA/SSM — более
  гранулярная реализация, чем виртуальный state NAViS; возможна только из-за `N_item ≪ N_user`.
- **H3/T5 (item-side достройка холодных)** имеет внешнюю опору (PTGCN +40.8% на холодных;
  DyGFormer/CCFCRec co-occurrence) — поднять приоритет.
- **Новизна** заявляема (плотная per-pair память не встречена), но честно — по аналогии/мотивации;
  нужен прямой head-to-head на TGB (наш T3/T9).

## Источники (первичные, peer-reviewed, если не отмечено)

| Работа | arXiv | Венью | Роль |
|---|---|---|---|
| TGNv2 (source-target id) | [2411.03596](https://arxiv.org/abs/2411.03596) | NeurIPS 2024 | базовый метод/проблема |
| TGB (node affinity, NDCG@10) | [2307.01026](https://arxiv.org/abs/2307.01026) | NeurIPS 2023 | задача/бенчмарк |
| NAViS (Virtual State / SSM) | [2510.06940](https://arxiv.org/abs/2510.06940) | preprint, Oct 2025 | конкурент/смежн. |
| JODIE (coupled RNN user/item) | [1908.01207](https://arxiv.org/abs/1908.01207) | KDD 2019 | per-entity state |
| TGSRec (CTBG + TCT) | [2108.06625](https://arxiv.org/abs/2108.06625) | CIKM 2021 | bipartite temporal |
| TGN-for-recsys | [2403.16066](https://arxiv.org/html/2403.16066v2) | preprint | bipartite temporal |
| DyGFormer (co-occurrence enc.) | [2303.13047](https://arxiv.org/abs/2303.13047) | NeurIPS 2023 | CF-подобный bias |
| CCFCRec (contrastive CF cold-start) | [2302.02151](https://arxiv.org/abs/2302.02151) | WWW 2023 | CF cold-start |
| PTGCN (cold-start, item dynamics) | [2107.05235](https://arxiv.org/abs/2107.05235) | ACM TOIS | ⭐ прямое подтверждение |
| TGAT (inductive, memoryless) | [2002.07962](https://arxiv.org/abs/2002.07962) | ICLR 2020 | cold-start ≠ per-pair |
| New-node few-shot link pred | [2310.09787](https://arxiv.org/abs/2310.09787) | preprint | cold-start ≠ per-pair |
| TGB-Seq | [2502.02975](https://arxiv.org/abs/2502.02975) | ICLR 2025 | bipartite bench/огранич. |
| Yu — TGB empirical eval | [2307.12510](https://arxiv.org/abs/2307.12510) | preprint | подтверждение эвристик |

*Метод: deep-research harness (fan-out поиск → fetch → 3-голосная adversarial-верификация,
убийство при ≥2/3 refute → синтез). Несколько формулировок «CF / warm item-side» — аналогии
синтеза, не дословные термины статей (атрибутировано выше).*
