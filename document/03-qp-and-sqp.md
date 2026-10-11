# 3. QP と SQP の使い方

ここでは、`sample/qp_active_set_demo_1.py` と `sample/sqp_2_mass_spring_damper_demo.py` を例に、静的な QP と制御問題向けの SQP を見ていきます。

## 3-1. `QP_ActiveSetSolver` は静的な二次計画問題に使う

QP の問題形は、一般に次のように書けます。

```text
minimize  (1/2) * x^T E x - L^T x
subject to M x <= gamma
```

`sample/qp_active_set_demo_1.py` はそのままこの形で書かれています。

```python
E = np.eye(2)
L = np.array([[5.0], [8.0]])
M = np.array([[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1]])

gamma = np.array([[4.0],
                  [6.0],
                  [0.0],
                  [0.0]])

solver = QP_ActiveSetSolver(number_of_variables=E.shape[0],
                            number_of_constraints=M.shape[0])
x_opt = solver.solve(E, L, M, gamma)
```

ここで大事なのは、すべてが「行列・ベクトルの形」でまとめられている点です。

- `E`: 二次項の行列
- `L`: 線形項のベクトル
- `M`: 制約条件の行列
- `gamma`: 制約の上限

これは、静的な最適化問題を簡単に書くための最も分かりやすい入力形式です。

### これが向いている問題

- 変数が固定されている
- 時間軸ごとの繰り返しがない
- 最適解がただ一度計算されればよい

## 3-2. `SQP_ActiveSet_PCG_PLS` は制御問題で使う

制御では、ただの QP ではなく、各時刻ごとに状態をシミュレートしながら、入力列全体を最適化していく必要があります。これが `SQP_ActiveSet_PCG_PLS` の出番です。

`sample/sqp_2_mass_spring_damper_demo.py` の中核は次の部分です。

```python
sqp_cost_matrices = SQP_CostMatrices_NMPC(
    x_syms=x_syms,
    u_syms=u_syms,
    state_equation_vector=f,
    measurement_equation_vector=h,
    Np=N,
    Qx=Qx,
    Qy=Qy,
    R=R,
    U_min=u_min,
    U_max=u_max,
)

solver = SQP_ActiveSet_PCG_PLS(U_size=(nu, N))

U_opt = solver.solve(
    U_horizon_initial=U_horizon_initial,
    cost_and_gradient_function=sqp_cost_matrices.compute_cost_and_gradient,
    cost_function=sqp_cost_matrices.compute_cost,
    hvp_function=sqp_cost_matrices.hvp_analytic,
    X_initial=X_initial,
    U_min_matrix=sqp_cost_matrices.U_min_matrix,
    U_max_matrix=sqp_cost_matrices.U_max_matrix,
)
```

### 役割の分担

- `SQP_CostMatrices_NMPC`: 状態方程式とコストをまとめて扱う
- `SQP_ActiveSet_PCG_PLS`: そのコストをもとに最適な入力列を反復的に更新する

この構成は、「制御問題の問題設定を簡単に作る」という観点では非常に便利です。

## 3-3. `SQP_CostMatrices_NMPC` の利点

`SQP_CostMatrices_NMPC` は、次のような情報をまとめて持つ役割を果たします。

- `x_syms`: 状態変数
- `u_syms`: 制御入力
- `f`: 状態遷移式
- `h`: 出力式
- `Qx`, `Qy`, `R`: コスト重み
- `U_min`, `U_max`: 入力制約
- `Np`: 予測ホライズン長

その結果、ユーザーは個別にコストや勾配を毎回手で書くのではなく、以下のように使えるようになります。

```python
J, gradient = sqp_cost_matrices.compute_cost_and_gradient(X_initial, U_horizon)
```

この関数群は、制御問題に特化した最適化問題を簡単に扱うためのラッパーとして働いています。

## 3-4. `SQP_ActiveSet_PCG_PLS` を理解するための視点

このクラスには、名前の通り次の3つの要素があります。

- Active Set: どの制約が今有効かを管理
- PCG: 連立方程式に対する効率的な反復計算
- PLS: 変数の上限/下限を守りながら段階的に改善

そのため、制御入力に上下限があるような大きめの最適化問題でも、探索を安全に進めることができます。

## 3-5. 使い分けの目安

| 問題 | 使うクラス | 使う理由 |
|---|---|---|
| 固定の QP | `QP_ActiveSetSolver` | 形式が簡単で、静的な最適化に向く |
| 予測制御・入力列を最適化 | `SQP_ActiveSet_PCG_PLS` | 制御の時間展開を扱える |
| その補助計算 | `SQP_CostMatrices_NMPC` | 遷移式・コスト・勾配を自動収集 |

## 3-6. 実践のコツ

- 最初は `qp_active_set_demo_1.py` のような小さなQ Pで入力形式に慣れる
- 次に `sqp_*_demo.py` のような制御サンプルに進む
- 最後に `SQP_CostMatrices_NMPC` の中で何を自動化しているかを見ると理解が深まる

このように、サンプルを段階的に読んでいくと、ライブラリの使い方が自然に見えてきます。
