# Optimisation for Machine Learning — Assignments

**Course:** CS7CS2 Optimisation for Machine Learning  
**Institution:** Trinity College Dublin, School of Computer Science and Statistics  
**Instructor:** Dr. Mouli Chakraborty

This repository contains five assignments completed for the *Optimisation for Machine Learning* module, covering foundational and advanced optimization algorithms used in machine learning. Each assignment includes executable Jupyter notebooks and question sheets.

---

## Assignment Overview

| Assignment | Topics | Key Algorithms |
|-----------|--------|----------------|
| [Week 2](./week2/) | Gradient Descent Fundamentals | Finite difference, GD, subgradient |
| [Week 4](./week4/) | Advanced Optimizers | Polyak, Heavy Ball, RMSProp, Adam |
| [Week 6](./week6/) | ML Optimization | Full-batch GD, Mini-batch SGD, NAG, Adagrad, Newton's method |
| [Week 8](./week8/) | Constrained Optimization | Projected GD, Penalty method, Primal-Dual, Frank-Wolfe |
| [Final](./Final/) | Comprehensive Final | Polyak, Adagrad, RMSprop, Heavy Ball, Nesterov, Adam, Newton, derivative-free, constrained |

---

### Week 2 — Gradient Descent Fundamentals

**Question sheet:** [`questions_week2.md`](./week2/questions_week2.md)  
**Notebook:** [`week2.ipynb`](./week2/week2.ipynb)

- **Q1:** SymPy symbolic differentiation, forward finite difference approximation of $f(x)=x^4$, MAE analysis vs. step size $\delta$, gradient descent implementation with multiple learning rates ($\alpha=0.05, 0.5, 1.2$)
- **Q2:** 2D contour visualization of $f(x)=0.5(x_1^2+10x_2^2)$, narrow valley effect of GD, local minima analysis on $f(x)=x^4-2x^2+0.1x$ with different starting points
- **Q3:** Learning rate analysis on $f(x)=x^2$ (convergence vs. divergence vs. oscillation), condition number $\gamma$ effect on convergence rate, subgradient method on $f(x)=|x|$ with oscillation near $x=0$
- **Q4:** Ill-conditioning on quadratic $f(x)=x_1^2+\gamma x_2^2$ with $\gamma=1,4$, GD on Rosenbrock function with $\alpha=0.001, 0.005$ over 2000 iterations

---

### Week 4 — Advanced Optimizers Comparison

**Question sheet:** [`ques.md`](./week4/ques.md)  
**Notebook:** [`week4.ipynb`](./week4/week4.ipynb)

- **Q1(I):** Comparison of Polyak step size, RMSProp ($\alpha=0.2,\beta=0.9$), Heavy Ball ($\alpha=0.01,\beta=0.9$), and Adam ($\alpha=0.1,\beta_1=0.9,\beta_2=0.999$) on $f(x,y)=x^2+100y^2$ with 200 iterations, log-scale convergence plots
- **Q1(II):** Heavy Ball learning rate sensitivity — tested $\alpha=0.006, 0.01, 0.02$; $\alpha=0.02$ diverges, demonstrating instability threshold
- **Q1(III):** Contour trajectory overlay of Heavy Ball, RMSProp, and Adam; analysis of zig-zag behavior, momentum smoothing, and adaptive preconditioning
- **Q2(I):** Same four algorithms on Rosenbrock banana function with 3000 iterations; Polyak step size capped at $\alpha\le0.1$, RMSProp ($\alpha=0.01$), Heavy Ball ($\alpha=2\times10^{-4}$), Adam ($\alpha=0.05$)
- **Q2(II):** Adam sensitivity analysis ($\alpha=0.02, 0.05, 0.12, 0.2, 0.3$); instability at $\alpha>0.12$
- **Q2(III):** Rosenbrock contour trajectories showing algorithm behavior along the curved valley

---

### Week 6 — Machine Learning Optimization

**Question sheet:** [`week6qs.md`](./week6/week6qs.md)  
**Notebook:** [`week6.ipynb`](./week6/week6.ipynb)

- **Q1(a):** Full-batch GD on linear regression $\hat{y}=X\theta$, $m=1000$, $\theta^*=[3,4]^T$, 80 iterations with $\alpha=0.5$, log-loss and contour trajectory plots
- **Q1(b):** Mini-batch SGD comparison $b=5$ vs $b=20$ over 400 updates; smaller batches yield noisier but faster convergence per-epoch
- **Q1(c):** NAG ($\beta=0.9$) and Adagrad vs constant-step SGD with $b=10$; Adagrad shows smoother convergence via per-coordinate scaling
- **Q2(a):** Full-batch GD on toy NN $\hat{y}=x_2\tanh(x_1u)$, $x^*=[1,3]^T$, 500 iterations with $\alpha=0.75$
- **Q2(b):** Mini-batch SGD on toy NN with $b=5$ and $b=20$ over 500 updates
- **Q2(c):** NAG vs Adagrad on toy NN with $b=10$
- **Q3(a):** GD on Rosenbrock with $\alpha=10^{-3}$, 200 iterations — slow linear convergence
- **Q3(b):** Newton's method with finite-difference Hessian, $\alpha=0.7$, 200 iterations — quadratic convergence
- **Q3(c):** Damped Newton (backtracking line search with contraction factor 0.5) — robust convergence comparison across all three methods

---

### Week 8 — Constrained Optimization

**Question sheet:** [`week8_q.md`](./week8/week8_q.md)  
**Notebook:** [`week8.ipynb`](./week8/week8.ipynb)

- **Q1:** Projected gradient descent on $f(x_1,x_2)=(x_1-1.2)^2+2(x_2-2.5)^2+0.4x_1x_2$ with two constraint sets:
  - $X_1$: box constraints $0.5\le x_1\le2.5$, $0.5\le x_2\le3.5$ (element-wise clipping)
  - $X_2$: $x_1\ge0.5$, $x_2\ge0.5$, $x_1\le x_2$ (repeated projection)
  - Convergence analysis: starting from $x^{(0)}=(2.4,0.7)$, final iterate location relative to boundary
- **Q2:** Penalty method vs primal-dual on $f(x_1,x_2)=(x_1-0.2)^2+(x_2-2)^2$ with $x_1\ge0.5$, $x_1x_2\ge1$:
  - Small penalty ($\lambda=0.5$) fails to enforce feasibility
  - Large penalty ($\lambda=4.0$) enforces constraints
  - Primal-dual ($\alpha=0.06,\beta=0.08$) with adaptive multipliers converges to feasible solution
- **Q3:** Frank-Wolfe algorithm on $f(x_1,x_2)=(x_1-1.5)^2+(x_2-1.2)^2$ with polyhedron constraints:
  - Vertex enumeration for linear subproblem
  - Step size $\gamma=0.8, 0.95, 0.1$ comparison (larger $\gamma$ accelerates convergence)
  - Comparison with projected gradient descent

---

### Final Assignment — Comprehensive Optimization

**Question sheet:** [`Final_q.md`](./Final/Final_q.md)  
**Notebook:** [`Final.ipynb`](./Final/Final.ipynb)

The final assignment integrates all concepts across six problems and three benchmark functions:

**Benchmarks:**
- **A:** Linear regression (quadratic loss), $m=1000$, $\theta^*=[3,4]^T$ — smooth strongly convex
- **B:** Toy NN with sin term $f(x)=(x_1-1)^2+5(x_2-2)^2+\sin(x_1)$ — non-quadratic, non-convex
- **C:** Rosenbrock function $f(x)=(1-x_1)^2+100(x_2-x_1^2)^2$ — ill-conditioned, narrow curved valley
- **D:** Box-constrained feasible set $X=\{0.5\le x_1\le5, -5\le x_2\le10\}$

**Q1 — Polyak, Adagrad, RMSprop, Heavy Ball:**
- Algorithm implementation with tuned hyperparameters per benchmark
- Convergence curves, contour trajectories, adaptive step-size evolution
- Dimensional comparison: convergence speed, smoothness, stability

**Q2 — Nesterov, Adam, Mini-batch SGD:**
- Nesterov accelerated gradient with adaptive momentum $\beta_k$
- Adam with bias-corrected first/second moment estimation
- Mini-batch SGD batch size comparison ($b=5$ vs $b=40$)
- Noise amplification experiment (6× label noise)

**Q3 — Newton's Method & Local Approximation:**
- 1D Taylor expansion of $g(x)=x^4$ at $x_0=0.25$ (first vs second order)
- Newton vs GD on all three benchmarks (GD: 80 iter, Newton: 20 iter)
- Update magnitude comparison per iteration

**Q4 — Derivative-Free Optimization:**
- Benchmark B: Forward finite-difference GD (good $\delta=0.05$ vs poor $\delta=0.8$), Nesterov random search
- Benchmark C: Nelder-Mead simplex, grid search $55\times55$

**Q5 — Constrained Optimization (Projected GD + Penalty):**
- Benchmark B with $x_1\ge0.5$, starting from infeasible $(0.2, 4.0)$
- Unconstrained GD violates constraint, projected GD enforces immediately
- Penalty method with $\lambda=0.15, 1.8, 4.5$ — larger $\lambda$ reduces violation
- Constraint violation analysis (full range + first 30 iterations zoom)

**Q6 — Linear Programming & Frank-Wolfe:**
- LP on rectangle with $f(x)=a^Tx$, $a=[1,2]^T$ — optimal at $(0.5, -5)$
- Frank-Wolfe for interior optimum $f(x)=(x_1-1)^2+(x_2-5)^2$ with $\beta=0.90, 0.985$
- Frank-Wolfe for boundary optimum $f(x)=x_1^2+x_2^2$ with $\beta=0.93$
- Evolution of $x_k$ and $z_k$ over iterations

---

## Repository Structure

```
E:/Data_Analysis/
├── .gitignore
├── README.md
├── week2/
│   ├── week2.ipynb          # Executable code
│   └── questions_week2.md   # Problem statements
├── week4/
│   ├── week4.ipynb
│   └── ques.md
├── week6/
│   ├── week6.ipynb
│   └── week6qs.md
├── week8/
│   ├── week8.ipynb
│   └── week8_q.md
└── Final/
    ├── Final.ipynb
    └── Final_q.md

---

*Generated for the CS7CS2 Optimisation for Machine Learning course at Trinity College Dublin.*
```
