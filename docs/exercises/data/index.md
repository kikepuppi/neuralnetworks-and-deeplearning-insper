---
exercise: data
ai_use: "I used Claude to draft a first version of the scripts and of this text, and to review my analysis. I then rewrote the code myself, ran every script, and checked each parameter, formula and number against the statement. I understand and can explain every part of what is submitted."
---

# Exercise 1: Data

> Due: Sep 10, 2026 (Thursday), commits until 23:59.

## Approach and reproducibility

One script per exercise. They share [`base.py`](https://github.com/kikepuppi/neuralnetworks-and-deeplearning-insper/blob/main/docs/exercises/data/code/base.py) with the seed, the paths and a helper that saves figures. Each script runs on its own from a clean clone:

```bash
python -m pip install -r requirements.txt
python docs/exercises/data/code/ex1_clouds.py
python docs/exercises/data/code/ex2_highdim.py
python docs/exercises/data/code/ex3_titanic.py
```

Every script starts its random state with `np.random.default_rng(42)` and uses that one generator until the end, so the numbers come out identical on every run. `scikit-learn` is used only for PCA and for the preprocessing.

The Kaggle file `train.csv` is committed under `docs/exercises/data/data/`.

## Exercise 1 
Point clouds: geometry and spread in 2D.

### A: Generate the clouds

400 samples, 100 per class. Each class is a Gaussian with its own mean and its own standard deviation.

| Class | Mean | Standard deviation | Mean spread `σ̄` |
|---|---|---|---|
| 0 | [2, 3] | [0.8, 2.5] | 1.65 |
| 1 | [5, 6] | [1.2, 1.9] | 1.55 |
| 2 | [8, 1] | [0.9, 0.9] | 0.90 |
| 3 | [15, 4] | [0.5, 2.0] | 1.25 |

![Figure 1](figures/fig1.png)

### B: More or less spread out

I regenerated the same four classes four times, multiplying every standard deviation by `s ∈ {0.5, 1.0, 2.0, 4.0}`. The means never move. These are four datasets of four classes each, not one dataset with four differently scaled classes.

![Figure 2](figures/fig2.png)

#### Separation ratio at `s = 1`

`r_ij = ‖μi - μj‖ / (σ̄i + σ̄j)`, with `σ̄k = (σ_kx + σ_ky) / 2`

| Pair | `‖μi - μj‖` | `σ̄i + σ̄j` | `r_ij` |
|---|---|---|---|
| (0, 1) | 4.2426 | 3.20 | **1.3258** |
| (0, 2) | 6.3246 | 2.55 | 2.4802 |
| (0, 3) | 13.0384 | 2.90 | 4.4960 |
| (1, 2) | 5.8310 | 2.45 | 2.3800 |
| (1, 3) | 10.1980 | 2.80 | 3.6422 |
| (2, 3) | 7.6158 | 2.15 | 3.5422 |

The smallest is the pair **(0, 1), at 1.3258**, which are the two clouds that visibly touch in Figure 1.

The means are fixed and only the spreads scale, so `r_ij(s) = r_ij(1) / s`. At `s = 2` the smallest ratio becomes **0.6629**, without generating any new points.

#### Mixing rate

The fraction of points whose nearest class center is not their own. The distance from each point to the four fixed means.

| `s` | Smallest `r_ij` | Mixing rate |
|---|---|---|
| 0.5 | 2.6516 | **0.0000** (0.00%) |
| 1.0 | 1.3258 | **0.0675** (6.75%) |
| 2.0 | 0.6629 | **0.2250** (22.50%) |
| 4.0 | 0.3315 | **0.4175** (41.75%) |

![Figure 3](figures/fig3.png)

#### From which scale can straight lines no longer separate the clouds?

From `s = 2` on.

At `s = 0.5` the mixing rate is exactly zero, and at `s = 1` it is 6.75%, almost all of it on the (0, 1) border. At `s = 2` it goes up to 22.50%, and that panel of Figure 2 shows classes 0, 1 and 2 as a single thing.

#### What happens to the smallest `r_ij` at that point?
At that same point the smallest `r_ij` drops below 1, to 0.6629. Below 1 means the two centers are closer to each other than the sum of their typical spreads, so the clouds overlap around their own midpoint and any line drawn between the centers has both classes on both sides of it.

### C: Analysis

#### Describe the overlap of the four classes at `s = 1`

Class 3 is on its own, nothing else gets near `x ≈ 15`. Class 2 is compact, low and a bit to the right, it touches class 1 only at its top. The real overlap is between classes 0 and 1, which is the pair with the smallest `r_ij`.

#### Could a single linear boundary separate all classes? What about a set of linear boundaries?

No. One boundary cuts the plane in two, so it only expresses a yes/no decision. Four classes need at least three boundaries.

#### Sketch on Figure 1 the decision boundaries you think a trained neural network might learn.

![Figure 1, annotated](figures/fig1_boundaries.png)

#### The more spread out the clouds are, what happens to the region where the network makes mistakes?

It gets bigger. The boundaries stay almost the same, but the classes overlap more. In this overlapping area, the network cannot clearly tell the classes apart. More training or data cannot fully remove these errors. The error rate grows from 0% at `s = 0.5` to 41.75% at `s = 4`.

### Code

```python
--8<-- "docs/exercises/data/code/base.py"
```

```python
--8<-- "docs/exercises/data/code/ex1_clouds.py"
```

## Exercise 2
Non-linearity in higher dimensions.

### A: Dataset I, shifted Gaussians

500 samples for class A and 500 for class B, from 5-dimensional multivariate normals with the mean vectors and covariance matrices given.

### B: Dataset II, concentric shells

500 samples per class, also in 5D. I draw `v ~ N(0, I₅)` and normalize it, `u = v / ‖v‖`, which gives a direction uniform on the unit sphere. Each point is `x = ρ · u`, with `ρ ~ N(2.0, 0.4)` for class C and `ρ ~ N(5.0, 0.4)` for class D.

### C: Visualize and compare

![Figure 4](figures/fig4.png)

#### Report the explained variance of the first two components in each case

| Dataset | PC1 | PC2 | PC1 + PC2 |
|---|---|---|---|
| I: shifted Gaussians | 0.5004 | 0.1593 | **0.6597** (65.97%) |
| II: concentric shells | 0.2159 | 0.2132 | **0.4291** (42.91%) |

#### In which dataset does the 2D projection better preserve the information relevant for classification?

Dataset I. There the direction with the most spread is also the direction that separates the classes, so PC1 alone holds half the variance.

Dataset II is isotropic, so the five components are all about the same size. The projection keeps 42.91% of the variance, and that part does not carry the label.

#### The distance between the class centers, in 5D

| Dataset | `‖μ1 - μ2‖` |
|---|---|
| I: shifted Gaussians | **3.2282** |
| II: concentric shells | **0.2662** |

Dataset II should be exactly zero, since both classes are centered on the origin. The 0.2662 is sampling noise.

#### Figure 5: the histogram of the radius of each point

![Figure 5](figures/fig5.png)

| Dataset | Class | Radius mean | Radius std |
|---|---|---|---|
| I | A | 2.1085 | 0.7869 |
| I | B | 4.1284 | 1.2248 |
| II | C (core) | 1.9718 | 0.3956 |
| II | D (shell) | 5.0047 | 0.4094 |

In Dataset II the two distributions do not touch. Class C goes from about 0.8 to 3.2, class D from 3.7 to 6.3.

### D: Analysis

#### In Dataset II the distance between the centers is close to zero, yet the radius histograms are well separated. What does that combination tell you about the possibility of separating the classes with a hyperplane?

It cannot be done. A hyperplane looks at one direction at a time. Both classes have the same center, so in any direction the two classes fall on top of each other with the same mean. Wherever you cut, half of each class stays on each side, so about 50%, the same as guessing.

#### Explain why the structure of Dataset II cannot be solved by a linear boundary, no matter how much data is collected.

The class depends only on the distance to the origin. That distance is the same if you rotate the space, so the rule does not care about direction. A hyperplane does care, it points somewhere, so it cannot copy the rule. More data does not fix that, the limit is the model.

#### Does a 2D projection in which the classes look mixed prove that they are inseparable?

No. PCA keeps the directions with the most spread, not the ones that separate the classes.

My own figure shows the opposite case. The right panel of Figure 4 is not mixed at all, it is a blue core inside a red ring, and still no straight line separates the colors.

#### A simple function of the inputs that separates Dataset II

`g(x) = Σᵢ xᵢ² - 12.25 = ‖x‖² - 3.5²`

The sign of `g` gives the class, negative for the core and positive for the shell. 3.5 is halfway between radius 2 and radius 5. With this one feature alone all 1000 points are right, 100.00%.

### Code

```python
--8<-- "docs/exercises/data/code/ex2_highdim.py"
```

## Exercise 3
Preparing real-world data for a neural network.

### A: Get to know the data

#### What does the `Transported` column represent? What is the class balance?

Whether the passenger was moved to another dimension by the anomaly. It is a boolean, so this is binary classification.

| `Transported` | Share |
|---|---|
| False | 49.64% |
| True | 50.36% |

#### List the features, separating numerical from categorical

| Type | Columns |
|---|---|
| Numerical | `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |
| Categorical | `HomePlanet`, `CryoSleep`, `Destination`, `VIP` |
| Identifier or free text (dropped) | `PassengerId`, `Cabin`, `Name` |

#### Table of missing values per column

| Column | Missing | Percent |
|---|---|---|
| CryoSleep | 217 | 2.50% |
| ShoppingMall | 208 | 2.39% |
| VIP | 203 | 2.34% |
| HomePlanet | 201 | 2.31% |
| Name | 200 | 2.30% |
| Cabin | 199 | 2.29% |
| VRDeck | 188 | 2.16% |
| FoodCourt | 183 | 2.11% |
| Spa | 183 | 2.11% |
| Destination | 182 | 2.09% |
| RoomService | 181 | 2.08% |
| Age | 179 | 2.06% |

#### Mean, median and maximum of the spending columns

| Column | Mean | Median | Maximum |
|---|---|---|---|
| RoomService | 224.69 | 0.00 | 14327.00 |
| FoodCourt | 458.08 | 0.00 | 29813.00 |
| ShoppingMall | 173.73 | 0.00 | 23492.00 |
| Spa | 311.14 | 0.00 | 22408.00 |
| VRDeck | 304.85 | 0.00 | 24133.00 |

#### Compare mean and median: what does that difference tell you about the spread and the skewness?

The median is zero in all five, but the means go from 174 to 458. So more than half the passengers spent nothing, and the mean comes from a few who spent a lot. The tails are heavy on the right. `FoodCourt` reaches 29813, about 65 times its own mean.

### B: Split before you transform

80/20, stratified by `Transported`, with a fixed seed. Done before any statistic is computed.

| Split | Rows | Positive share |
|---|---|---|
| Train | 6954 | 0.5036 |
| Test | 1739 | 0.5037 |

#### Why this split comes before imputation and scaling

The median, the minimum, the maximum and the list of categories are all learned from data. If they are computed on the full dataset, the test rows leak into the transformation used on the training rows. The test score then measures data the pipeline has already seen, which makes it look better than it is.

### C: Preprocess

#### Missing data: strategy per column type

Fitted on the training set and applied to the test set.

| Type | Strategy | Value learned on train | Why |
|---|---|---|---|
| `Age` | median | 27.0 | The distribution is skewed, and the median is not pulled by the tail. |
| Spending columns | median | 0.0 for all five | The median is 0, which is what most passengers do. The mean would invent spending for people we know nothing about. |
| Categorical | most frequent | Earth, False, TRAPPIST-1e, False | Keeps the column categorical without a "missing" level learned from 200 rows. |

#### Convert the categorical features to numerical format

`HomePlanet`, `CryoSleep`, `Destination` and `VIP` become 10 binary columns with one-hot encoding:

`HomePlanet_Earth`, `HomePlanet_Europa`, `HomePlanet_Mars`, `CryoSleep_False`, `CryoSleep_True`, `Destination_55 Cancri e`, `Destination_PSO J318.5-22`, `Destination_TRAPPIST-1e`, `VIP_False`, `VIP_True`

One-hot instead of integer codes, because integer codes create an order that does not exist.

#### Explain how your code handles a category that appears in the test set but not in the training set

With `OneHotEncoder(handle_unknown="ignore")`. The encoder is fitted on the training set, so its columns are fixed there. An unseen category becomes zero in every column of that feature, instead of an error or an extra column.

#### Create the `TotalSpend` feature

The sum of the five spending columns, computed before the log. `Cabin`, `Name` and `PassengerId` are dropped.

#### Apply `log(1 + x)` to the spending columns

Applied to the five spending columns and to `TotalSpend`. `log1p` and not `log` because the columns have real zeros.

![Figure 6](figures/fig6.png)

#### Why does this transformation help a network with `tanh`?

Raw `FoodCourt` goes from 0 to 29813, with half the values at exactly 0. After a linear rescale almost everyone sits at -1 and a few outliers take the rest of the range. `tanh` is flat outside about `[-2, 2]`, so those outliers have almost no gradient, and the rest of the column barely varies. After the log the column goes from 0 to 10.3 and the values spread out, so they land where `tanh` still has gradient.

#### Scaling: implement one, explain the choice, and report the minimum and maximum

The seven numerical columns are normalized to `[-1, 1]` with `MinMaxScaler`, fitted on the training set.

I chose normalization because `tanh` outputs `[-1, 1]`. Mapping the training range onto that puts every input where the network can learn. Standardization leaves the extremes free, and after the log `FoodCourt` still has a tail.

| Set | Minimum | Maximum |
|---|---|---|
| Train (numerical columns) | -1.0000 | 1.0000 |
| Test (numerical columns) | -1.0000 | **1.1383** |

The test set goes past 1 because someone there spent 23492 on `ShoppingMall`, while the training maximum was 12253. The scaler never saw the test set, so it could not know. If the test range were exactly `[-1, 1]`, that would be the suspicious result.

### D: Verify and visualize

Figure 6 above shows `FoodCourt` before and after the log, with the count axis on a log scale.

#### Final checks

| Check | Train | Test |
|---|---|---|
| Remaining `NaN` | **0** | **0** |
| Shape of the feature matrix | **(6954, 17)** | **(1739, 17)** |
| Value range | [-1.0000, 1.0000] | [-1.0000, 1.1383] |

The 17 columns are 7 numerical plus 10 one-hot. The range fits `tanh`, which needs inputs inside about `[-2, 2]`.

#### Which of your preprocessing decisions would most affect the network's training, and why?

The `log(1 + x)` on the spending columns. The other choices change little: a different imputation touches 2% of the rows, and min-max or standardization both keep the inputs in a range `tanh` handles. Without the log, five of the seven numerical columns would be almost constant after scaling, with half the passengers at the floor and a few outliers alone at the top. The network would get five inputs with almost no gradient, and would have to classify with `Age` and the categorical columns only. The spending columns are the most predictive part of this dataset, so the log is what makes them usable.

### Code

```python
--8<-- "docs/exercises/data/code/ex3_titanic.py"
```

## Results summary

| # | Item | Value |
|---|---|---|
| 1 | Mixing rate at `s = 0.5` | 0.0000 (0.00%) |
| 2 | Mixing rate at `s = 1.0` | 0.0675 (6.75%) |
| 3 | Mixing rate at `s = 2.0` | 0.2250 (22.50%) |
| 4 | Mixing rate at `s = 4.0` | 0.4175 (41.75%) |
| 5 | Smallest `r_ij` at `s = 1.0`, and which pair | 1.3258, pair (0, 1) |
| 6 | Distance between centers, Dataset I | 3.2282 |
| 7 | Distance between centers, Dataset II | 0.2662 |
| 8 | Explained variance PC1 + PC2, Dataset I | 0.6597 (65.97%) |
| 9 | Explained variance PC1 + PC2, Dataset II | 0.4291 (42.91%) |
| 10 | Share of the positive class in `Transported` | 0.5036 (50.36%) |
| 11 | Mean and median of `FoodCourt` on the training set, before transforming | mean 452.6112, median 0.0000 |
| 12 | Final shape of the training feature matrix | (6954, 17) |
| 13 | Minimum and maximum after scaling | train [-1.0000, 1.0000], test [-1.0000, 1.1383] |
