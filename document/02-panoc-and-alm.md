# 2. 制約付き最適化の基本：PANOC と ALM/PM

この章では、`sample/panoc_demo.py` と `sample/alm_pm_optimizer_demo.py` の2つを軸にして、クラスの使い分けを説明します。

## 2-1. `PANOC_Optimizer` は「箱制約に向く」

`PANOC_Optimizer` は、各変数に上下限がある問題を扱うのに向いています。特徴は次の通りです。

- 目的関数は `cost_func(u)`
- 勾配は `gradient_func(u)`
- 制約は `u_min`, `u_max`
- `solve(u0)` で最適化を実行

```python
cache = PANOC_Cache(problem_size=n, tolerance=1e-8, lbfgs_memory=10)

solver = PANOC_Optimizer(
    cost_func=cost,
    gradient_func=gradient,
    cache=cache,
    u_min=u_min,
    u_max=u_max,
    max_iteration=200,
)

status = solver.solve(u0)
```

この形で使うと、ソルバは内部的に次のような手順を意識して動きます。

- 勾配を計算する
- 変数を制約内に射影する
- 近似 Hessian を使って方向を決める
- 収束判定を行う

つまり、ユーザーが一からアルゴリズムを書かなくても、`cost_func` と `gradient_func` をきちんと定義しておけば、箱型の inequality 制約を自動で処理してくれます。

### 使いどころ

- `u_min <= u <= u_max` が自然にある
- 変数ごとに上下限がある
- 非線形なコスト関数を扱いたい

## 2-2. `ALM_PM_Optimizer` は「工夫した制約を追加したい」場合に向く

一方で、`ALM_PM_Optimizer` は制約が単なる上下限ではなく、出力の条件や集合条件で表されるときに使いやすいです。`sample/alm_pm_optimizer_demo.py` では、次のような制約がありました。

```text
u0 + u1 <= 1.0
```

これは box 制約ではなく、変数の結合に対する不等式です。そこで、次のように制約関数を分離します。

```python
def F1(u):
    return np.array([[u[0, 0] + u[1, 0]]])


def project_C(x):
    np.minimum(x, 1.0, out=x)
```

このとき、`ALM_Factory` は内部で次のような拡張コストを組み立てます。

```text
psi(u; xi) = f(u) + (c/2) * dist_C(F1(u) + y/c)^2
```

ここで `xi = (c, y)` は、罰則パラメータ `c` と Lagrange multiplier `y` をまとめたものです。ユーザーがこれを計算式で意識しなくても、`ALM_Factory` が `psi(u; xi)` と `d_psi(u; xi)` を作ってくれます。

### `ALM_Factory` の役割

`ALM_Factory` は、最適化が使える形に「増強したコスト」と「勾配」を変換する役割を持っています。

- `f`: 元のコスト
- `df`: 元の勾配
- `mapping_f1`: 制約の出力関数
- `jacobian_f1_trans`: 制約の Jacobian の転置積
- `set_c_project`: 制約集合への射影

これを渡して `factory.psi(...)` と `factory.d_psi(...)` を作り、その結果を `ALM_Problem` に載せます。

## 2-3. 使うときの流れ

`ALM_PM_Optimizer` の利用フローは、概ね次のようになります。

```python
factory = ALM_Factory(...)
problem = ALM_Problem(
    parametric_cost=factory.psi,
    parametric_gradient=factory.d_psi,
    u_min=u_min,
    u_max=u_max,
    mapping_f1=F1,
    set_c_project=project_C,
    set_y_project=project_Y_operator.project,
    n1=1,
)

alm_cache = ALM_Cache(PANOC_Cache(...), n1=1)
optimizer = ALM_PM_Optimizer(
    alm_cache=alm_cache,
    alm_problem=problem,
    initial_penalty=10.0,
)

optimizer.solve(u0)
```

ここで見ておきたいのは、`PANOC_Optimizer` と比較して、`ALM_PM_Optimizer` は「制約の種類を増やせる」点です。

- `PANOC_Optimizer`: 箱型制約に向いている
- `ALM_PM_Optimizer`: 制約集合や不等式制約に向いている

## 2-4. どちらを選べばよいか

| 問題の形 | 向いているクラス | 例 |
|---|---|---|
| 各変数が上下限を持つ | `PANOC_Optimizer` | `-1.5 <= u <= 1.5` |
| 出力や複合条件に制約がある | `ALM_PM_Optimizer` | `u0 + u1 <= 1` |

## 2-5. 実際にサンプルを読むときのコツ

- `panoc_demo.py` は「箱型制約をどう扱うか」を見に行くサンプル
- `alm_pm_optimizer_demo.py` は「制約関数をどうクラスに渡すか」を見に行くサンプル

この2つを対比すると、リポジトリで扱う最適化の入口がかなり見えやすくなります。

次は、静的な QP と制御問題向け SQP について見ていきます。
