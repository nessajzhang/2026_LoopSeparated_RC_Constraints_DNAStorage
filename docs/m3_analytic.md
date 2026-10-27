# Exact analytic DNA m=3 spectral product

For every subset `U` of the 64 DNA triples,

\[
\rho(U)\rho(U^\perp)\le 9,
\qquad U^\perp=\Sigma^3\setminus\operatorname{RC}_\theta(U).
\]

The choice `U={A,C,G}^3` attains equality. All 64 bits are independent:
there is no RC-free or 32-orbit restriction.

The global bound comes from the human-readable analytic theorem below.
The reproducible certificate checks exact finite identities and the
equality witness. It is neither enumeration of `2^64` configurations nor
a proof-assistant formalization. Previous incomplete search certificates
remain separate incomplete evidence; their frontiers are not closed by
relabeling them.

## Reproduction

From the code checkout, using its existing Python 3.12 environment:

```sh
uv run --no-sync python code/certify_m3_analytic.py --output-dir /absolute/path/to/new-attempt
```

One command produces the certificate and invokes the independent checker
in an isolated Python subprocess. The output directory must not exist:
attempts are append-only, never overwritten. It records the proof,
configuration, environment, public-source hashes, certificate, checker
result, logs, resource summary, and output hashes. The certificate itself
is deterministic; timings and environment records need not be identical
across machines.

The independent checker can also be run alone:

```sh
uv run --no-sync python -I code/check_m3_analytic_certificate.py --certificate /absolute/path/to/new-attempt/certificate.json
```

It imports only the Python standard library, never the producer module,
and uses no eigensolver. It checks all single-edge bases of the RC
transpose identity, integer witness matrices and eigenvectors, three
characteristic polynomials, the elimination polynomial, zero-product
and reducible zero-inner-product cases, and all 256 binary `q=2,m=3`
partitions with exact rational SCC bounds. A locked proof-text hash
detects changes requiring mathematical re-review; a hash is not itself
a proof of the theorem.

```python
from loop_alpha_audit.spectral_m3_analytic import analytic_certificate

data = analytic_certificate()  # JSON-ready, deterministic; does not write files
```

Tests and local style checks:

```sh
uv run --no-sync pytest tests/test_m3_analytic.py tests/test_m3_certificates.py tests/test_regimes.py
uv run --no-sync ruff check --no-respect-gitignore loop_alpha_audit/spectral_m3_analytic.py code/certify_m3_analytic.py code/check_m3_analytic_certificate.py tests/test_m3_analytic.py
uv run --no-sync ruff format --check --no-respect-gitignore loop_alpha_audit/spectral_m3_analytic.py code/certify_m3_analytic.py code/check_m3_analytic_certificate.py tests/test_m3_analytic.py
```

## General-q analytic theorem for m=3

Let `D` be the row-to-column overlap matrix on ordered pairs over an
alphabet of size `q>=2`, with `D[(a,b),(b,c)]=1`. For any nonnegative
matrices `A+B=D`, including fractional edge partitions,

\[
\rho(A)+\rho(B)\le\frac{3q}{2},\qquad
\rho(A)\rho(B)\le\frac{9q^2}{16}.
\]

Every ordered pair of states has exactly one two-step path, so `D^2=J`.
Every row of `D` sums to `q`. Write `lambda=rho(A)`, `mu=rho(B)`.
If either radius vanishes, monotonicity gives their sum at most `q`.
Otherwise choose nonnegative nonzero Perron vectors
`Ax=lambda x` and `p^T B=mu p^T`. They need not be strictly positive,
so reducible matrices are included. Define

\[
X=\sum_{a,b}x_{ab},\quad Y=\sum_{a,b}p_{ab},\quad
h_b=\sum_c x_{bc},\quad k_b=\sum_a p_{ab},\quad
S=\lambda+\mu,\quad z=\lambda\mu,
\]

\[
T=\frac{p^TDx}{XY}=\sum_b\frac{h_b}{X}\frac{k_b}{Y}.
\]

Since `lambda x_ab<=h_b`, summing over `b` gives `lambda h_a<=X`.
The left Perron equation similarly gives `mu p_bc<=k_b` and
`mu k_c<=Y`. Thus `u_b=lambda h_b/X`, `v_b=mu k_b/Y` are in `[0,1]`,
with sums `lambda,mu`. Summing `u_b v_b>=u_b+v_b-1` yields

\[
T\ge\frac{S-q}{z}. \tag{1}
\]

For the reverse bound let `d=Dx/lambda-x=Bx/lambda>=0`. The Perron
equations give `p^T d=mu p^T x/lambda` and `p^T Dx=S p^T x`.
Multiplying `mu p^T<=p^T D` by `d>=0` and using `D^2=J` gives

\[
\mu p^Td\le p^TDd=\frac{XY}{\lambda}-p^TDx.
\]

Substitution gives

\[
T\le\frac{S}{\lambda^2+\lambda\mu+\mu^2}
=\frac{S}{S^2-z}. \tag{2}
\]

No step divides by `p^T x`, which may vanish. If `S<=q`, the desired
sum bound is immediate. Otherwise (1) and (2) imply
`(S-q)S^2<=(2S-q)z`. Because `z<=S^2/4` and `2S-q>0`, this gives
`S-q<=(2S-q)/4`, hence `S<=3q/2`. The product bound follows from
`z<=S^2/4`. The exact elimination identity is

\[
4\{(S-q)S^2-(2S-q)\lambda\mu\}
=S^2(2S-3q)+(2S-q)(\lambda-\mu)^2.
\]

This proof uses the specific one-letter marginal caps of the order-two
overlap graph as well as `D^2=J`. No general-m analogue is asserted.

## RC bridge and correctly labelled DNA witness

For any alphabet involution, let `R(a,b)=(theta(b),theta(a))`. An edge
`i->j` of `S` becomes `R(j)->R(i)` under RC, so

\[
M_{\operatorname{RC}_\theta(S)}=R M_S^T R^{-1}.
\]

Because `Uperp=RC(complement U)`, its spectral radius equals that of
the ordinary complement. Thus the theorem bounds the manuscript's
actual objective, not a substituted language.

For `U={A,C,G}^3`, the matrix's maximum row sum is 3, and the vector
equal to 1 on states in `{A,C,G}^2` and 0 elsewhere is a nonzero
eigenvector of eigenvalue 3. Therefore its radius is exactly 3.

The ordinary complement consists of all triples containing T. In
contrast, the actual `Uperp` consists of all triples containing A.
For the latter use the strictly positive **right** vector equal to
3 on states ending in A, 2 on states beginning in A but not ending
in A, and 1 elsewhere. Direct multiplication gives eigenvalue 3.
The ordinary complement uses the analogous right vector with T.
The full characteristic polynomials are

\[
\chi_U(t)=t^{15}(t-3),\qquad
\chi_{U^\perp}(t)=\chi_{\Sigma^3\setminus U}(t)
=t^{13}(t^3-t^2-3t-9).
\]

The cubic factors as `(t-3)(t^2+2t+3)`. Together with the global
theorem, this proves the exact maximum 9.

For the upper-linear branch the resulting exact asymptotic DNA
input-constraint capacity is

\[
C_3(\alpha)=2(2\alpha-1)+(1-\alpha)\log_2 9,
\qquad \tfrac12\le\alpha\le1.
\]

This is not a full finite codebook formula, an IDS channel capacity,
a noisy-decoder result, a FER guarantee, or a biochemical advantage.

## Attribution

The deficit-vector idea is adapted from Lemma 12 of
[Agarwal et al., arXiv:2607.15941v2](https://arxiv.org/abs/2607.15941),
PDF pp. 5–6. That lemma treats symmetric matrices summing to `J`; it
does not state the directed `m=3` overlap theorem above. The present
argument adds a left Perron vector, `D^2=J`, and the one-letter caps.

Verified preprint bibliography item; no journal publication is asserted:

```tex
\bibitem{agarwal2026nordhaus}
S. Agarwal, C. Antley, J. Aulenbacher, G. Brooks, I. Gonzalez,
L. Hawranick, W. Linz, L. Lu, and A. Williams,
``Generalized Nordhaus--Gaddum inequalities for eigenvalues,''
arXiv:2607.15941v2 [math.CO], 2026.
```
