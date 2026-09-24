# EXPLORE.md — MAP: Charting Student Math Misunderstandings

> 纯探查报告，未做任何建模 / 训练 / API 调用。`data/` 下文件未被修改。
> 生成方式：pandas 读取 `data/train.csv`。

## TL;DR — 6 个关键事实

1. **规模**：36,696 行 × 7 列，只有 **15 道题**，平均每题 2,446 条作答。
2. **标签空间**：6 个 `Category` × 35 个 `Misconception`，实际组合 **65 种**。`Misconception` 有 73% 是 NaN（写作 `NA`）。
3. **极度不均衡**：Top-3 组合（全是 `*:NA`）占 72.5%，Top-20 占 95.4%；最大类 14,802 条 vs 最小类 1 条。
4. **长尾在组合层、不在裸标签层**：裸 `Misconception` 最少也有 6 条（1–5 次的有 0 个）；但 `Category:Misconception` 组合里有 14 种 ≤5 条，几乎全在 `True_Misconception:*` 这一支。
5. **文本很短**：`StudentExplanation` 中位 13 词、p90 26 词、最长 177 词——短文本分类，长上下文模型没什么用武之地。
6. **两个坑**：`Wrong_fraction` / `Wrong_Fraction` 只差大小写却是两个类；61 组完全相同的作答被打了不同标签（标注噪声上限）。


## 1. train.csv 基本结构

- 行数：**36,696**
- 列数：**7**

| # | 列名 | dtype | 非空数 | 缺失数 | 唯一值数 |
|---|------|-------|--------|--------|----------|
| 0 | `row_id` | int64 | 36,696 | 0 | 36,696 |
| 1 | `QuestionId` | int64 | 36,696 | 0 | 15 |
| 2 | `QuestionText` | str | 36,696 | 0 | 15 |
| 3 | `MC_Answer` | str | 36,696 | 0 | 49 |
| 4 | `StudentExplanation` | str | 36,696 | 0 | 35,726 |
| 5 | `Category` | str | 36,696 | 0 | 6 |
| 6 | `Misconception` | str | 9,860 | 26,836 | 35 |

```
<class 'pandas.DataFrame'>
RangeIndex: 36696 entries, 0 to 36695
Data columns (total 7 columns):
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   row_id              36696 non-null  int64
 1   QuestionId          36696 non-null  int64
 2   QuestionText        36696 non-null  str  
 3   MC_Answer           36696 non-null  str  
 4   StudentExplanation  36696 non-null  str  
 5   Category            36696 non-null  str  
 6   Misconception       9860 non-null   str  
dtypes: int64(2), str(5)
memory usage: 2.0 MB
```

## 2. 标签列长什么样

### 2.1 Category 取值分布

| Category | 条数 | 占比 |
|----------|------|------|
| `True_Correct` | 14,802 | 40.34% |
| `False_Misconception` | 9,457 | 25.77% |
| `False_Neither` | 6,542 | 17.83% |
| `True_Neither` | 5,265 | 14.35% |
| `True_Misconception` | 403 | 1.10% |
| `False_Correct` | 227 | 0.62% |

共 **6** 种 Category。

### 2.2 Misconception 取值分布

- `Misconception` 列缺失（NaN，表示无误解标签）：**26,836** 条（73.13%）
- 非缺失：**9,860** 条（26.87%）
- 不同的非空 Misconception 取值：**35** 种

完整取值列表（含 NaN），按条数降序：

| Misconception | 条数 | 占比 |
|---------------|------|------|
| NaN (无误解) | 26,836 | 73.13% |
| `Incomplete` | 1,454 | 3.96% |
| `Additive` | 929 | 2.53% |
| `Duplication` | 704 | 1.92% |
| `Subtraction` | 620 | 1.69% |
| `Positive` | 566 | 1.54% |
| `Wrong_term` | 558 | 1.52% |
| `Irrelevant` | 497 | 1.35% |
| `Wrong_fraction` | 418 | 1.14% |
| `Inversion` | 414 | 1.13% |
| `Mult` | 353 | 0.96% |
| `Denominator-only_change` | 336 | 0.92% |
| `Whole_numbers_larger` | 329 | 0.90% |
| `Adding_across` | 307 | 0.84% |
| `WNB` | 299 | 0.81% |
| `Tacking` | 290 | 0.79% |
| `Unknowable` | 282 | 0.77% |
| `Wrong_Fraction` | 273 | 0.74% |
| `SwapDividend` | 206 | 0.56% |
| `Scale` | 179 | 0.49% |
| `Not_variable` | 154 | 0.42% |
| `Firstterm` | 107 | 0.29% |
| `Adding_terms` | 97 | 0.26% |
| `Multiplying_by_4` | 96 | 0.26% |
| `FlipChange` | 78 | 0.21% |
| `Division` | 63 | 0.17% |
| `Definition` | 54 | 0.15% |
| `Interior` | 50 | 0.14% |
| `Longer_is_bigger` | 24 | 0.07% |
| `Ignores_zeroes` | 23 | 0.06% |
| `Shorter_is_bigger` | 23 | 0.06% |
| `Base_rate` | 23 | 0.06% |
| `Inverse_operation` | 21 | 0.06% |
| `Certainty` | 18 | 0.05% |
| `Incorrect_equivalent_fraction_addition` | 9 | 0.02% |
| `Wrong_Operation` | 6 | 0.02% |

### 2.3 组合标签 Category:Misconception

- 组合标签总数：**65** 种（这就是比赛实际要预测的类别空间）
- 最多的一类 `True_Correct:NA`：14,802 条（40.34%）
- 最少的一类 `True_Misconception:Base_rate`：1 条（0.0027%）
- 头尾比：**14,802 : 1**
- 只出现 1 次的组合：**5** 种；≤5 次：**14** 种；≤10 次：**24** 种
- 前 10 类累计占比：**86.63%**
- 前 20 类累计占比：**95.35%**
- 后 45 类（共 1,706 条）合计仅占 **4.65%**

完整组合分布（降序）：

| # | Category:Misconception | 条数 | 占比 | 累计占比 |
|---|------------------------|------|------|----------|
| 1 | `True_Correct:NA` | 14,802 | 40.337% | 40.34% |
| 2 | `False_Neither:NA` | 6,542 | 17.828% | 58.16% |
| 3 | `True_Neither:NA` | 5,265 | 14.348% | 72.51% |
| 4 | `False_Misconception:Incomplete` | 1,446 | 3.940% | 76.45% |
| 5 | `False_Misconception:Additive` | 891 | 2.428% | 78.88% |
| 6 | `False_Misconception:Duplication` | 698 | 1.902% | 80.78% |
| 7 | `False_Misconception:Subtraction` | 618 | 1.684% | 82.47% |
| 8 | `False_Misconception:Positive` | 564 | 1.537% | 84.00% |
| 9 | `False_Misconception:Wrong_term` | 550 | 1.499% | 85.50% |
| 10 | `False_Misconception:Wrong_fraction` | 412 | 1.123% | 86.63% |
| 11 | `False_Misconception:Irrelevant` | 409 | 1.115% | 87.74% |
| 12 | `False_Misconception:Inversion` | 409 | 1.115% | 88.85% |
| 13 | `False_Misconception:Mult` | 345 | 0.940% | 89.79% |
| 14 | `False_Misconception:Denominator-only_change` | 332 | 0.905% | 90.70% |
| 15 | `False_Misconception:Whole_numbers_larger` | 328 | 0.894% | 91.59% |
| 16 | `False_Misconception:Adding_across` | 306 | 0.834% | 92.43% |
| 17 | `False_Misconception:WNB` | 291 | 0.793% | 93.22% |
| 18 | `False_Misconception:Unknowable` | 282 | 0.768% | 93.99% |
| 19 | `False_Misconception:Wrong_Fraction` | 273 | 0.744% | 94.73% |
| 20 | `False_Correct:NA` | 227 | 0.619% | 95.35% |
| 21 | `False_Misconception:SwapDividend` | 198 | 0.540% | 95.89% |
| 22 | `False_Misconception:Scale` | 179 | 0.488% | 96.38% |
| 23 | `True_Misconception:Tacking` | 162 | 0.441% | 96.82% |
| 24 | `False_Misconception:Not_variable` | 153 | 0.417% | 97.24% |
| 25 | `False_Misconception:Tacking` | 128 | 0.349% | 97.59% |
| 26 | `False_Misconception:Adding_terms` | 97 | 0.264% | 97.85% |
| 27 | `False_Misconception:Firstterm` | 96 | 0.262% | 98.11% |
| 28 | `False_Misconception:Multiplying_by_4` | 93 | 0.253% | 98.36% |
| 29 | `True_Misconception:Irrelevant` | 88 | 0.240% | 98.60% |
| 30 | `False_Misconception:FlipChange` | 74 | 0.202% | 98.81% |
| 31 | `False_Misconception:Division` | 58 | 0.158% | 98.96% |
| 32 | `False_Misconception:Definition` | 51 | 0.139% | 99.10% |
| 33 | `False_Misconception:Interior` | 50 | 0.136% | 99.24% |
| 34 | `True_Misconception:Additive` | 38 | 0.104% | 99.34% |
| 35 | `False_Misconception:Longer_is_bigger` | 23 | 0.063% | 99.41% |
| 36 | `False_Misconception:Ignores_zeroes` | 23 | 0.063% | 99.47% |
| 37 | `False_Misconception:Base_rate` | 22 | 0.060% | 99.53% |
| 38 | `False_Misconception:Inverse_operation` | 21 | 0.057% | 99.59% |
| 39 | `False_Misconception:Certainty` | 18 | 0.049% | 99.63% |
| 40 | `True_Misconception:Shorter_is_bigger` | 17 | 0.046% | 99.68% |
| 41 | `True_Misconception:Firstterm` | 11 | 0.030% | 99.71% |
| 42 | `True_Misconception:Incomplete` | 8 | 0.022% | 99.73% |
| 43 | `True_Misconception:WNB` | 8 | 0.022% | 99.75% |
| 44 | `True_Misconception:SwapDividend` | 8 | 0.022% | 99.78% |
| 45 | `True_Misconception:Mult` | 8 | 0.022% | 99.80% |
| 46 | `True_Misconception:Wrong_term` | 8 | 0.022% | 99.82% |
| 47 | `False_Misconception:Incorrect_equivalent_fraction_addition` | 7 | 0.019% | 99.84% |
| 48 | `True_Misconception:Duplication` | 6 | 0.016% | 99.86% |
| 49 | `False_Misconception:Wrong_Operation` | 6 | 0.016% | 99.87% |
| 50 | `False_Misconception:Shorter_is_bigger` | 6 | 0.016% | 99.89% |
| 51 | `True_Misconception:Wrong_fraction` | 6 | 0.016% | 99.90% |
| 52 | `True_Misconception:Inversion` | 5 | 0.014% | 99.92% |
| 53 | `True_Misconception:Division` | 5 | 0.014% | 99.93% |
| 54 | `True_Misconception:FlipChange` | 4 | 0.011% | 99.94% |
| 55 | `True_Misconception:Denominator-only_change` | 4 | 0.011% | 99.95% |
| 56 | `True_Misconception:Definition` | 3 | 0.008% | 99.96% |
| 57 | `True_Misconception:Multiplying_by_4` | 3 | 0.008% | 99.97% |
| 58 | `True_Misconception:Incorrect_equivalent_fraction_addition` | 2 | 0.005% | 99.98% |
| 59 | `True_Misconception:Subtraction` | 2 | 0.005% | 99.98% |
| 60 | `True_Misconception:Positive` | 2 | 0.005% | 99.99% |
| 61 | `True_Misconception:Not_variable` | 1 | 0.003% | 99.99% |
| 62 | `True_Misconception:Whole_numbers_larger` | 1 | 0.003% | 99.99% |
| 63 | `True_Misconception:Longer_is_bigger` | 1 | 0.003% | 99.99% |
| 64 | `True_Misconception:Adding_across` | 1 | 0.003% | 100.00% |
| 65 | `True_Misconception:Base_rate` | 1 | 0.003% | 100.00% |

### 2.4 Category × Misconception 交叉（是否有效组合）

每个 Category 下出现的不同 Misconception 数量：

| Category | 不同 Misconception 数（含 NaN） | 是否带真实误解标签 |
|----------|-------------------------------|--------------------|
| `False_Correct` | 1 | 否 |
| `False_Misconception` | 35 | 是（35 种） |
| `False_Neither` | 1 | 否 |
| `True_Correct` | 1 | 否 |
| `True_Misconception` | 26 | 是（26 种） |
| `True_Neither` | 1 | 否 |

## 3. Misconception 长尾

- 不同 Misconception（不含 NaN）：**35** 个
- 带 Misconception 标签的样本：**9,860** 条

### 3.1 最常见的前 20 个 Misconception

| # | Misconception | 条数 | 占全体 train | 占有误解样本 |
|---|---------------|------|--------------|--------------|
| 1 | `Incomplete` | 1,454 | 3.96% | 14.75% |
| 2 | `Additive` | 929 | 2.53% | 9.42% |
| 3 | `Duplication` | 704 | 1.92% | 7.14% |
| 4 | `Subtraction` | 620 | 1.69% | 6.29% |
| 5 | `Positive` | 566 | 1.54% | 5.74% |
| 6 | `Wrong_term` | 558 | 1.52% | 5.66% |
| 7 | `Irrelevant` | 497 | 1.35% | 5.04% |
| 8 | `Wrong_fraction` | 418 | 1.14% | 4.24% |
| 9 | `Inversion` | 414 | 1.13% | 4.20% |
| 10 | `Mult` | 353 | 0.96% | 3.58% |
| 11 | `Denominator-only_change` | 336 | 0.92% | 3.41% |
| 12 | `Whole_numbers_larger` | 329 | 0.90% | 3.34% |
| 13 | `Adding_across` | 307 | 0.84% | 3.11% |
| 14 | `WNB` | 299 | 0.81% | 3.03% |
| 15 | `Tacking` | 290 | 0.79% | 2.94% |
| 16 | `Unknowable` | 282 | 0.77% | 2.86% |
| 17 | `Wrong_Fraction` | 273 | 0.74% | 2.77% |
| 18 | `SwapDividend` | 206 | 0.56% | 2.09% |
| 19 | `Scale` | 179 | 0.49% | 1.82% |
| 20 | `Not_variable` | 154 | 0.42% | 1.56% |

### 3.2 长尾有多长

**直接回答：只出现 1–5 次的 Misconception 共 0 个。** 最稀有的 Misconception 是 `Wrong_Operation`，出现 6 次——也就是说，在 **裸 Misconception 名称** 这个层面上并不存在真正的长尾：全部 35 个标签每个至少出现 6 次。

| 出现次数 | Misconception 个数 | 占全部 Misconception 种类 | 覆盖样本数 |
|----------|--------------------|---------------------------|------------|
| = 1 | 0 | 0.0% | 0 |
| = 2 | 0 | 0.0% | 0 |
| = 3 | 0 | 0.0% | 0 |
| = 4 | 0 | 0.0% | 0 |
| = 5 | 0 | 0.0% | 0 |
| **1–5（小计）** | 0 | 0.0% | 0 |
| 6–10 | 2 | 5.7% | 15 |
| 11–50 | 7 | 20.0% | 182 |
| 51–100 | 5 | 14.3% | 388 |
| > 100 | 21 | 60.0% | 9,275 |

出现次数最少的 10 个 Misconception：

| Misconception | 条数 |
|---------------|------|
| `Wrong_Operation` | 6 |
| `Incorrect_equivalent_fraction_addition` | 9 |
| `Certainty` | 18 |
| `Inverse_operation` | 21 |
| `Ignores_zeroes` | 23 |
| `Shorter_is_bigger` | 23 |
| `Base_rate` | 23 |
| `Longer_is_bigger` | 24 |
| `Interior` | 50 |
| `Definition` | 54 |

### 3.3 长尾真正在哪里：组合标签层面

长尾出现在 **`Category:Misconception` 组合** 上，而不是裸标签上——因为 `True_Misconception:*` （学生选对了答案、但解释里暴露了误解）这一支整体只有 403 条，摊到 26 种 Misconception 上，大多每种只剩个位数。

| 组合出现次数 | 组合个数 | 覆盖样本数 |
|--------------|----------|------------|
| = 1 | 5 | 5 |
| **1–5** | 14 | 35 |
| 6–10 | 10 | 71 |
| 11–50 | 9 | 223 |
| 51–100 | 7 | 557 |
| > 100 | 25 | 35,810 |

其中 ≤5 次的 14 种组合里，**14 种属于 `True_Misconception:*`**。

## 4. StudentExplanation 长度分布

### 4.1 按空格分词的词数

| 统计量 | 词数 | 字符数 |
|--------|------|--------|
| min | 1.0 | 1.0 |
| p10 | 7.0 | 33.0 |
| p25 | 10.0 | 43.0 |
| median (p50) | 13.0 | 60.0 |
| mean | 15.4 | 70.0 |
| p75 | 19.0 | 86.0 |
| p90 | 26.0 | 120.0 |
| p95 | 31.0 | 145.0 |
| p99 | 43.0 | 205.0 |
| max | 177.0 | 586.0 |

- 空字符串 / 缺失的 StudentExplanation：**0** 条
- 词数 ≤ 3 的：**51** 条（0.14%）
- 词数 ≥ 50 的：**147** 条（0.40%）

### 4.2 最短的 5 条（原文）

**#1** — row_id=30752, QuestionId=89443, 词数=1, 字符数=1, 标签=`True_Neither:NA`

```text
'd'
```

**#2** — row_id=29403, QuestionId=89443, 词数=1, 字符数=2, 标签=`False_Neither:NA`

```text
'??'
```

**#3** — row_id=30133, QuestionId=89443, 词数=1, 字符数=2, 标签=`True_Neither:NA`

```text
'-3'
```

**#4** — row_id=30776, QuestionId=89443, 词数=1, 字符数=2, 标签=`True_Neither:NA`

```text
'd.'
```

**#5** — row_id=35669, QuestionId=109465, 词数=1, 字符数=2, 标签=`False_Neither:NA`

```text
'b.'
```

### 4.3 最长的 1 条

row_id=30422, QuestionId=89443, 词数=177, 字符数=521, 标签=`True_Neither:NA`

（原文已删除：比赛规则禁止再分发数据。）

## 5. 题目（QuestionId）

- 不同题目数：**15**
- 平均每题作答数：**2446.4** 条
- 中位数：**2610**，min=**673**，max=**4857**
- p10=**1105**, p25=**1654**, p75=**3080**, p90=**3430**
- 不同 QuestionText：**15**（与 QuestionId 一致）
- 不同 MC_Answer 字符串：**49**
- 每题不同选项数：min=**4**, median=**4**, max=**4**

作答数最多的 10 道题：

| QuestionId | 作答数 | 不同选项数 | 不同 Misconception 数 |
|------------|--------|------------|----------------------|
| 31772 | 4857 | 4 | 2 |
| 31778 | 3640 | 4 | 3 |
| 31774 | 3115 | 4 | 3 |
| 32833 | 3105 | 4 | 3 |
| 89443 | 3054 | 4 | 2 |
| 31777 | 2809 | 4 | 3 |
| 33472 | 2800 | 4 | 3 |
| 91695 | 2610 | 4 | 2 |
| 32835 | 2332 | 4 | 4 |
| 32829 | 2156 | 4 | 3 |

作答数最少的 10 道题：

| QuestionId | 作答数 | 不同选项数 | 不同 Misconception 数 |
|------------|--------|------------|----------------------|
| 104665 | 673 | 4 | 2 |
| 109465 | 1051 | 4 | 2 |
| 76870 | 1186 | 4 | 3 |
| 33471 | 1542 | 4 | 2 |
| 33474 | 1766 | 4 | 2 |
| 32829 | 2156 | 4 | 3 |
| 32835 | 2332 | 4 | 4 |
| 91695 | 2610 | 4 | 2 |
| 33472 | 2800 | 4 | 3 |
| 31777 | 2809 | 4 | 3 |

- 每题出现的不同 Misconception 数：min=**2**, median=**3**, max=**4**；完全没有 Misconception 标签的题目：**0** 道

## 6. 随机 10 条完整样本

原本此处列出 10 条随机样本（`random_state=42`）的全部字段原文。比赛规则禁止再分发数据，已删除。
要查看原始样本，请自行从 Kaggle 下载 `train.csv`，用 `random_state=42` 重新抽样。

## 7. 数据质量观察（探查中撞到的坑）

### 7.1 Misconception 名称存在大小写重复

- `Wrong_Fraction` vs `Wrong_fraction`：拼写只差大小写。
  - `Wrong_Fraction`：273 条，只出现在 QuestionId 31777
  - `Wrong_fraction`：418 条，只出现在 QuestionId 33471

  两者被当作**不同类别**计入 65 种组合。因为分别绑在不同题目上，合并与否会直接影响类别数与每题候选集，建模前需要明确决定（合并 → 组合数减 1）。

### 7.2 重复作答与标签冲突

- `StudentExplanation` 文本重复的行：**970** 条
- (QuestionId, MC_Answer, StudentExplanation) 三元组完全重复的行：**797** 条，分布在 **580** 个重复组里
- 其中标签（`Category:Misconception`）**不一致**的重复组：**61** 个（占 10.5%）

  即同样的题 + 同样的选项 + 一字不差的解释，被打上了不同标签——标注噪声的天花板，CV 分数解读时要记得。

  几个冲突例子（原文已删除，只列标签）：

| QuestionId | 冲突的两个标签 | 示例中的冲突组数 |
|------------|----------------|------------------|
| 31772 | `True_Correct:NA` / `True_Neither:NA` | 3 |
| 31772 | `True_Correct:NA` / `True_Misconception:Incomplete` | 2 |

### 7.3 每道题只用到组合标签空间的一小块

- 全局有 65 种组合标签，但单题实际出现的组合只有 **6–11** 种（中位 8）

| QuestionId | 作答数 | 该题出现的组合标签数 |
|------------|--------|----------------------|
| 31772 | 4,857 | 8 |
| 31774 | 3,115 | 10 |
| 31777 | 2,809 | 7 |
| 31778 | 3,640 | 9 |
| 32829 | 2,156 | 8 |
| 32833 | 3,105 | 9 |
| 32835 | 2,332 | 11 |
| 33471 | 1,542 | 7 |
| 33472 | 2,800 | 10 |
| 33474 | 1,766 | 8 |
| 76870 | 1,186 | 8 |
| 89443 | 3,054 | 8 |
| 91695 | 2,610 | 8 |
| 104665 | 673 | 8 |
| 109465 | 1,051 | 6 |

  只有 15 道题，且题目在 train/test 间大概率重叠，**按 QuestionId 限制候选标签集**是显然的先验。但也意味着模型极易过拟合到这 15 道题，对新题的泛化无法用本地 CV 衡量。

## 附录：test.csv / sample_submission.csv

- `test.csv`：**3** 行 × **5** 列，列名：`row_id`, `QuestionId`, `QuestionText`, `MC_Answer`, `StudentExplanation`
  （公开的只是占位样本；实际测试集在提交时替换。没有 `Category` / `Misconception` 列。）
- `sample_submission.csv`：**3** 行 × **2** 列，列名：`row_id`, `Category:Misconception`
- 提交格式：每行 `row_id` + 一个 `Category:Misconception` 字符串，**空格分隔最多 3 个预测**（MAP@3）。示例：

```text
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
36697,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
36698,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
```

