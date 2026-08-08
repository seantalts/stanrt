# stanli

**The Stan Language Interpreter.** Compile and sample Stan models with no
C++ toolchain on the machine.

[![PyPI](https://img.shields.io/pypi/v/stanli.svg)](https://pypi.org/project/stanli/)
[![Python](https://img.shields.io/pypi/pyversions/stanli.svg)](https://pypi.org/project/stanli/)
[![License](https://img.shields.io/pypi/l/stanli.svg)](https://github.com/seantalts/stanli/blob/main/LICENSE)
[![wheels](https://github.com/seantalts/stanli/actions/workflows/wheels.yml/badge.svg)](https://github.com/seantalts/stanli/actions/workflows/wheels.yml)

```console
pip install stanli
```

That is the whole install. No compiler, no `make`, no CmdStan checkout, no
multi-minute first-run build. One wheel, one shared library, under seven
megabytes.

```python
import stanli

model = stanli.Model(stan_file="eight_schools.stan", data="data.json")
draws = model.sample(seed=1, warmup=1000, samples=1000)

draws["mu"].mean()      # one numpy array of draws per constrained parameter
```

Model preparation takes milliseconds, so the first draw arrives about 20x
sooner than a toolchain that compiles C++ per model.

## How it works

Every Stan model is a composition of a fixed vocabulary of operations:
densities, constraint transforms, linear algebra, elementwise math. stanli
ships those precompiled and turns each model into *data*, a static graph of
ops over flat preallocated buffers, instead of generating and compiling C++
per model.

```
model.stan + data.json
  |  stanc3, the official OCaml compiler, linked into the library
  v
transformed MIR
  |  lowering: transformed data evaluated eagerly, data-bound loops unrolled,
  |            then periodic regions re-rolled back into vectorized ops
  v
op graph over preallocated value/adjoint arenas
  |  forward sweep = log density, reverse sweep = gradient
  v
NUTS with diagonal-metric adaptation -> draws
```

The graph doubles as the autodiff tape, so a reverse sweep is a backwards
loop over an array rather than a walk through a pointer-chasing tape, and
steady-state gradient evaluation allocates nothing.

Two things are not reimplemented, which is what makes the results
trustworthy: the compiler is the real stanc3, linked in-process, so the
Stan language behaves as the official toolchain makes it behave; and the
math is unmodified stan-math, the same code CmdStan runs.

## Correctness

Nothing here ships on "looks close".

**<!--gen:corpus_verified_of-->118 of 120<!--/gen--> posteriordb models**
are differentially verified against CmdStan: same model, same data, same
evaluation point, comparing the log density and every single gradient
component. **<!--gen:corpus_bitwise-->45<!--/gen--> agree bitwise.** The
worst deviation across the entire corpus is
**<!--gen:corpus_worst-->2.6e-12<!--/gen--> relative**.

The two exceptions are documented rather than hidden. `sir`'s ODE solution
dips about 1e-9 below a declared lower bound at the shared evaluation point,
where CmdStan rejects it too; `kronecker_gp` matches on the log density and
436 of 438 gradients, differing on the two that flow through eigenvectors of
a nearly degenerate covariance matrix.

Full per-model accuracy table:
[docs/corpus-status.md](https://github.com/seantalts/stanli/blob/main/docs/corpus-status.md)

## Performance

Per-gradient latency against CmdStan, same models, same evaluation point,
both sides `-O3` with FP contraction pinned off:

<!--gen:bench_table_us-->
| model | params | stanli | CmdStan | speedup |
| --- | ---: | ---: | ---: | ---: |
| `radon_pooled` | 3 | 52.9 us | 320.9 us | **6.1x** |
| `arK` | 7 | 2.4 us | 12.5 us | **5.2x** |
| `radon_hierarchical_intercept_centered` | 391 | 111.6 us | 569.1 us | **5.1x** |
| `radon_county_intercept` | 388 | 89.7 us | 431.6 us | **4.8x** |
| `nes` | 10 | 19.7 us | 69.3 us | **3.5x** |
| `eight_schools_noncentered` | 10 | 0.23 us | 0.74 us | **3.3x** |
| `election88_full` | 90 | 295.3 us | 902.0 us | **3.0x** |
| `bym2_offset_only` | 3845 | 39.6 us | 114.6 us | **2.9x** |
| `dogs` | 3 | 22.0 us | 63.7 us | **2.9x** |
| `kidscore_momiq` | 3 | 1.9 us | 4.9 us | **2.6x** |
| `lsat_model` | 1006 | 45.5 us | 91.2 us | **2.0x** |
| `state_space_stochastic_level_stochastic_seasonal` | 389 | 17.2 us | 26.3 us | **1.5x** |
| `normal_mixture` | 3 | 79.0 us | 88.2 us | **1.1x** |
| `low_dim_gauss_mix` | 5 | 88.9 us | 98.3 us | **1.1x** |
| `wells_dist100ars_model` | 3 | 17.4 us | 19.0 us | **1.1x** |
| `radon_county` | 389 | 83.2 us | 82.1 us | **1.0x** |
| `arma11` | 4 | 6.7 us | 6.2 us | 0.93x |
| `diamonds` | 26 | 35.4 us | 31.5 us | 0.89x |
| `garch11` | 4 | 11.2 us | 9.7 us | 0.86x |
| `hmm_drive_0` | 6 | 173.0 us | 132.8 us | 0.77x |
| `hmm_example` | 4 | 36.3 us | 27.1 us | 0.75x |
| `ldaK2` | 7 | 145.9 us | 104.1 us | 0.71x |
| `iohmm_reg` | 29 | 545.2 us | 320.3 us | 0.59x |
<!--/gen-->

The wins come from op granularity. CmdStan's var tape allocates, walks, and
frees one node per scalar operation per leapfrog step; stanli pays a fixed
cost per *op*, and a vectorized statement over N elements amortizes that to
nothing. Across the whole posteriordb corpus the median is
<!--gen:corpus_median-->2.07x<!--/gen--> and
<!--gen:corpus_at_par-->93<!--/gen--> of
<!--gen:corpus_n_grad-->119<!--/gen--> models are at or above CmdStan.

The losses are honest and understood, and they are all one shape: a
recurrence. `hmm_*`, `garch11` and `arma11` step through time with each
step reading the last one's parameter-dependent result, which nothing can
vectorize, so the work is scalar on both sides and CmdStan's generated C++
is the faster way to run scalar work. `ldaK2` is a mixture over more than
two components, which the fusion pass does not yet widen.

ODE models are the other place stanli is still behind. An ODE right-hand
side is the one user function that cannot be inlined at lowering time,
since the integrator picks the times; it now compiles into a flat register
machine instead of being tree-walked, and the forward sweep keeps the
sensitivities it was already computing instead of solving twice. Together
that is 29x to 39x faster than the tree-walking interpreter it replaces,
which puts `lotka_volterra` and `soil_incubation` at 0.58x and 0.63x of
CmdStan rather than 0.015x.

Method and full table:
[docs/benchmarks.md](https://github.com/seantalts/stanli/blob/main/docs/benchmarks.md)

## API

The surface is small on purpose.

```python
import stanli

# A path to a .stan file, or the model source directly.
model = stanli.Model(stan_file="model.stan", data="data.json")
model = stanli.Model(stan_code=src, data={"J": 8, "y": y, "sigma": sigma})

model.n_unconstrained               # length of the unconstrained vector
model.constrained_names             # ['mu', 'tau', 'theta.1', ...]

lp, grad = model.log_prob_grad(q)   # sampling log density and its gradient

draws = model.sample(seed=1, warmup=1000, samples=1000, delta=0.8)
draws["mu"]                         # ndarray of length `samples`
```

`data` accepts a path to a JSON file or a dict of Python scalars, lists, and
numpy arrays. `sample` returns one array of constrained draws per scalar
parameter, named the way CmdStan names them, so `theta` declared as
`vector[8]` arrives as `theta.1` through `theta.8`.

## Platforms

Wheels for macOS (arm64 and x86_64), Linux (x86_64 and aarch64,
manylinux_2_28) and Windows (x86_64). The Windows wheel is built under
mingw-w64, because stan-math does not build under MSVC, which is the same
reason RStan ships through RTools. It bundles `stanc.exe` and runs it as
a subprocess instead of embedding the compiler, which waits on opam's
native Windows support; `Model("model.stan", data)` works the same way
either way.

The installed library is 22.2 MB, which is the trade this design makes:
ship the compiler and every kernel once, so that nothing is ever built on
the user's machine. Roughly half of that is the embedded stanc3 and
somewhat under half is stan-math. The interpreter and NUTS together are
about 410 KB.

## Status

Early, and deliberately narrow. The sampler is Stan's own NUTS with
diagonal-metric adaptation. Known limits, stated plainly:

- `sample()` returns declared parameters only. Transformed parameters and
  generated quantities are computed by the runtime and written by the
  command line tool, but are not exposed through the Python API yet, so
  the non-centered eight schools gives you `mu`, `tau`, and
  `theta_tilde`, not `theta`.
- No variational inference, no optimization, no multi-chain threading.
- No convergence diagnostics. Pair it with ArviZ or similar for now.

What is here is verified against CmdStan model by model, and every number
on this page is reproducible from the repository.

- Source, issues, and roadmap:
  [github.com/seantalts/stanli](https://github.com/seantalts/stanli)
- License: BSD-3-Clause, matching Stan's own.
