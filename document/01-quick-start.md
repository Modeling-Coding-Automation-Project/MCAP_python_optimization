# 1. はじめに：サンプルコードの読み方

このリポジトリのサンプルは、最適化クラスの「使い方」を学ぶための入口として作られています。各ファイルには、以下の流れがほぼ共通して入っています。

1. 問題の定義
2. コスト関数と勾配の定義
3. 制約や境界条件の定義
4. キャッシュまたはソルバの生成
5. 初期推定値を入れて `solve(...)` を呼ぶ

## 1-1. 最もシンプルな例：`panoc_demo.py`

`sample/panoc_demo.py` は、ボックス制約付きの非線形最適化を行う最も分かりやすい例です。

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

u0 = np.array([[-1.0], [2.0]])
status = solver.solve(u0)
```

ここで必要なのは、最小限の 4 つの要素です。

- `cost_func`: 目的関数
- `gradient_func`: グラディエント
- `u_min`, `u_max`: 各変数の上下限
- `u0`: 初期推定値

この形を覚えておくと、他の例も同じように理解しやすくなります。

## 1-2. 次の一歩：`alm_pm_optimizer_demo.py`

制約が「1 つの出力に対する不等式」になる場合は `ALM_PM_Optimizer` を使います。こちらは `PANOC_Optimizer` の「箱型制約付き」問題に、Lagrange multiplier に相当する変数を足した形に近い考え方です。

```python
factory = ALM_Factory(
    f=cost,
    df=gradient,
    mapping_f1=F1,
    jacobian_f1_trans=JF1_trans,
    set_c_project=project_C,
    n1=1,
)

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

optimizer = ALM_PM_Optimizer(
    alm_cache=alm_cache,
    alm_problem=problem,
    epsilon_tolerance=1e-5,
    delta_tolerance=1e-4,
    initial_penalty=10.0,
)

optimizer.solve(u0)
```

この例のポイントは「制約を関数と射影に分離して扱う」ことです。

- `F1(u)`: 制約の評価関数
- `JF1_trans`: その Jacobian の転置を掛ける関数
- `project_C`: 制約集合 `C` への射影
- `project_Y_operator`: Lagrange multiplier を定義域に収める射影

つまり、制約条件をそのまま関数で書くのではなく、「制約を評価して、許容集合に戻す」作業をクラス側に任せているのが見て取れます。

## 1-3. QP の例：`qp_active_set_demo_1.py`

QP は、最適化問題が二次関数の形に固定されている場合に使います。

```python
solver = QP_ActiveSetSolver(number_of_variables=E.shape[0],
                            number_of_constraints=M.shape[0])

x_opt = solver.solve(E, L, M, gamma)
```

ここで与えるのは次のような要素です。

- `E`: 二次項の Hessian
- `L`: 線形項
- `M`: 制約行列
- `gamma`: 制約右辺

問題の形は以下の通りです。

```text
minimize  (1/2) * x^T E x - L^T x
subject to M x <= gamma
```

この例は「最適化ソルバを使うときの入力形式を理解する」には非常に良いです。

## 1-4. 制御問題向けの例：`sqp_2_mass_spring_damper_demo.py`

制御・予測制御の問題では、ただの QP ではなく、状態遷移とコストを毎回更新する SQP が必要になります。サンプルでは以下のように組み立てます。

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

ここで重要なのは、`SQP_CostMatrices_NMPC` が「制御問題のコスト・勾配・Hessian をまとめて提供する」役割を担っている点です。`SQP_ActiveSet_PCG_PLS` はその結果を使って入力系列を改善していきます。

## 1-5. まず覚えるべき視点

最初のうちは、クラス名よりも次の 3 つを覚えると読みやすくなります。

- 「目的関数は何か」
- 「どんな制約を入れるか」
- 「初期値と境界は何を与えるか」

この 3 つが揃えば、サンプルコードはかなり読みやすくなります。

次のページでは、`PANOC_Optimizer` と `ALM_PM_Optimizer` の違いを、箱制約と出力制約の観点から見ていきます。
