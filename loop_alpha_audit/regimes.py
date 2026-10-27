"""Exact spectral-regime certificates for integer directed multigraphs.

No spectral floating-point calculation is used here.  In particular, parallel
edges remain parallel edges: the order-zero overlap matrix is ``[[len(S)]]``.
Vertex/word orders are lexicographic over ``range(q)`` throughout.
"""

from __future__ import annotations

from collections import deque
from itertools import product
from numbers import Integral
from typing import Iterable, Sequence

Matrix = tuple[tuple[int, ...], ...]
Word = tuple[int, ...]


def integer_matrix(adjacency: Sequence[Sequence[int]]) -> Matrix:
    rows = tuple(tuple(row) for row in adjacency)
    n = len(rows)
    if n < 1 or any(len(row) != n for row in rows):
        raise ValueError("the adjacency matrix must be nonempty and square")
    if any(not isinstance(v, Integral) or v < 0 for row in rows for v in row):
        raise ValueError("adjacency entries must be nonnegative integers")
    return tuple(tuple(int(v) for v in row) for row in rows)


def strongly_connected_components(
    adjacency: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Nonrecursive Kosaraju implementation, retaining isolated vertices."""

    matrix = integer_matrix(adjacency)
    n = len(matrix)
    edges = tuple(tuple(j for j, value in enumerate(row) if value) for row in matrix)
    reverse = tuple(tuple(i for i in range(n) if matrix[i][j]) for j in range(n))
    seen: set[int] = set()
    order: list[int] = []
    for start in range(n):
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, 0)]
        while stack:
            v, index = stack[-1]
            if index == len(edges[v]):
                order.append(v)
                stack.pop()
                continue
            w = edges[v][index]
            stack[-1] = (v, index + 1)
            if w not in seen:
                seen.add(w)
                stack.append((w, 0))
    assigned: set[int] = set()
    components = []
    for start in reversed(order):
        if start in assigned:
            continue
        assigned.add(start)
        stack = [start]
        component = []
        while stack:
            v = stack.pop()
            component.append(v)
            for w in reverse[v]:
                if w not in assigned:
                    assigned.add(w)
                    stack.append(w)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _return_path(matrix: Matrix, start: int, end: int, component: set[int]) -> list[int]:
    parents: dict[int, int | None] = {start: None}
    queue = deque([start])
    while end not in parents:
        v = queue.popleft()
        for w in sorted(component):
            if matrix[v][w] and w not in parents:
                parents[w] = v
                queue.append(w)
    path = [end]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    return list(reversed(path))


def classify_graph(adjacency: Sequence[Sequence[int]]) -> dict:
    """Return exact ``zero``, ``unit``, or ``positive`` and a finite witness.

    ``positive`` means radius strictly greater than one, not merely nonzero.
    Only *internal* SCC outdegrees enter the test.  Edges joining SCCs may
    change polynomial prefactors but cannot change the spectral regime.
    """

    matrix = integer_matrix(adjacency)
    components = strongly_connected_components(matrix)
    cyclic = []
    branch = None
    for component in components:
        members = set(component)
        degrees = [sum(matrix[v][w] for w in component) for v in component]
        has_cycle = len(component) > 1 or bool(matrix[component[0]][component[0]])
        if not has_cycle:
            continue
        cyclic.append(component)
        for v, degree in zip(component, degrees):
            if degree <= 1 or branch is not None:
                continue
            outgoing = [(v, w, label) for w in component for label in range(matrix[v][w])]
            closed_walks = []
            for edge in outgoing[:2]:
                path = _return_path(matrix, edge[1], v, members)
                walk = [list(edge)]
                walk.extend([a, b, 0] for a, b in zip(path, path[1:]))
                closed_walks.append(walk)
            branch = {"component": list(component), "vertex": v, "closed_walks": closed_walks}
    regime = "positive" if branch else "unit" if cyclic else "zero"
    result = {
        "schema": "integer-multigraph-regime-v1",
        "adjacency": [list(row) for row in matrix],
        "components": [list(c) for c in components],
        "regime": regime,
        "regime_index": {"zero": 0, "unit": 1, "positive": 2}[regime],
    }
    if branch:
        result["branch"] = branch
    elif cyclic:
        cycles = []
        for component in cyclic:
            start = component[0]
            cycle = [start]
            v = start
            while True:
                v = next(w for w in component if matrix[v][w])
                cycle.append(v)
                if v == start:
                    break
            cycles.append(cycle)
        result["cycles"] = cycles
    else:
        indegrees = [sum(row[j] for row in matrix) for j in range(len(matrix))]
        queue = deque(i for i, value in enumerate(indegrees) if not value)
        topological = []
        while queue:
            v = queue.popleft()
            topological.append(v)
            for w, value in enumerate(matrix[v]):
                if value:
                    indegrees[w] -= value
                    if not indegrees[w]:
                        queue.append(w)
        result["topological_order"] = topological
    return result


def verify_graph_certificate(certificate: dict) -> bool:
    """Independent reachability-matrix verification of a regime certificate.

    This deliberately does not call the producer's SCC or classification
    functions.  Invalid, inconsistent, and incomplete witnesses raise ValueError.
    """

    matrix = integer_matrix(certificate["adjacency"])
    n = len(matrix)
    reach = [[bool(matrix[i][j]) or i == j for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    reach[i][j] |= reach[k][j]
    remaining = set(range(n))
    components = []
    while remaining:
        v = min(remaining)
        comp = sorted(w for w in remaining if reach[v][w] and reach[w][v])
        components.append(comp)
        remaining.difference_update(comp)
    if certificate.get("components") != components:
        raise ValueError("incorrect SCC partition")
    cycles = [c for c in components if len(c) > 1 or matrix[c[0]][c[0]]]
    branching = any(sum(matrix[v][w] for w in c) > 1 for c in cycles for v in c)
    regime = "positive" if branching else "unit" if cycles else "zero"
    if certificate.get("regime") != regime:
        raise ValueError("incorrect spectral regime")
    if certificate.get("regime_index") != {"zero": 0, "unit": 1, "positive": 2}[regime]:
        raise ValueError("incorrect spectral regime index")
    if regime == "zero":
        order = certificate.get("topological_order", [])
        if sorted(order) != list(range(n)):
            raise ValueError("missing topological vertices")
        positions = {v: i for i, v in enumerate(order)}
        if any(matrix[v][w] and positions[v] >= positions[w] for v in range(n) for w in range(n)):
            raise ValueError("invalid topological order")
    elif regime == "unit":
        supplied = certificate.get("cycles", [])
        if len(supplied) != len(cycles):
            raise ValueError("missing cycle")
        for component, walk in zip(cycles, supplied):
            if not walk or walk[0] != walk[-1] or sorted(walk[:-1]) != component:
                raise ValueError("cycle does not cover its SCC once")
            if any(matrix[a][b] != 1 for a, b in zip(walk, walk[1:])):
                raise ValueError("invalid simple-cycle edge")
    else:
        branch = certificate.get("branch", {})
        component = branch.get("component", [])
        v = branch.get("vertex")
        walks = branch.get("closed_walks", [])
        if component not in cycles or v not in component or len(walks) != 2:
            raise ValueError("missing branching SCC witness")
        if not walks[0] or not walks[1] or walks[0][0] == walks[1][0]:
            raise ValueError("the two closed walks must have distinct first labelled edges")
        for walk in walks:
            current = v
            for edge in walk:
                if len(edge) != 3:
                    raise ValueError("invalid labelled edge")
                a, b, label = edge
                if a != current or a not in component or b not in component:
                    raise ValueError("discontinuous closed walk")
                if not isinstance(label, int) or not 0 <= label < matrix[a][b]:
                    raise ValueError("nonexistent parallel edge")
                current = b
            if current != v:
                raise ValueError("walk is not closed")
    return True


def word_index(word: Sequence[int], q: int) -> int:
    value = 0
    for symbol in word:
        if not isinstance(symbol, Integral) or not 0 <= symbol < q:
            raise ValueError("symbol is outside the alphabet")
        value = q * value + int(symbol)
    return value


def overlap_matrix(q: int, m: int, core: Iterable[Sequence[int]]) -> Matrix:
    if q < 2 or m < 1:
        raise ValueError("require q>=2 and m>=1")
    words = {tuple(word) for word in core}
    size = q ** (m - 1)
    rows = [[0] * size for _ in range(size)]
    for word in words:
        if len(word) != m:
            raise ValueError("core contains a word of the wrong length")
        word_index(word, q)
        rows[word_index(word[:-1], q)][word_index(word[1:], q)] += 1
    return tuple(tuple(row) for row in rows)


def higher_block_core(q: int, m: int, core: Iterable[Sequence[int]]) -> frozenset[Word]:
    words = frozenset(tuple(word) for word in core)
    overlap_matrix(q, m, words)  # validate all words
    return frozenset(word + (a,) for word in words for a in range(q) if word[1:] + (a,) in words)


def higher_block_factors(q: int, m: int, core: Iterable[Sequence[int]]) -> tuple[Matrix, Matrix]:
    """Rectangular P,Q with PQ=M_S, QP=M_T, including m=1."""

    words = frozenset(tuple(word) for word in core)
    overlap_matrix(q, m, words)
    small, large = q ** (m - 1), q**m
    p = [[0] * large for _ in range(small)]
    qq = [[0] * small for _ in range(large)]
    for word in words:
        edge = word_index(word, q)
        p[word_index(word[:-1], q)][edge] = 1
        qq[edge][word_index(word[1:], q)] = 1
    return tuple(map(tuple, p)), tuple(map(tuple, qq))


def maximal_rc_free_cores(q: int, m: int, theta: Sequence[int]):
    theta = tuple(theta)
    if (
        len(theta) != q
        or sorted(theta) != list(range(q))
        or any(theta[theta[a]] != a for a in range(q))
    ):
        raise ValueError("theta is not an involution of range(q)")
    pairs = []
    for word in product(range(q), repeat=m):
        partner = tuple(theta[a] for a in reversed(word))
        if word < partner:
            pairs.append((word, partner))
    for mask in range(1 << len(pairs)):
        yield mask, frozenset(pair[(mask >> i) & 1] for i, pair in enumerate(pairs))


def involution_regime(theta: Sequence[int], m: int) -> dict:
    """Proved classification of max_{S RC-free} rho(S), for fixed q,m,theta."""

    q = len(theta)
    if (
        q < 2
        or m < 1
        or sorted(theta) != list(range(q))
        or any(theta[theta[a]] != a for a in range(q))
    ):
        raise ValueError("invalid involution or stem length")
    f = sum(theta[a] == a for a in range(q))
    p = (q - f) // 2
    if m == 1:
        regime = "zero" if p == 0 else "unit" if p == 1 else "positive"
    elif p >= 2 or (p == 1 and f >= 1):
        regime = "positive"
    elif p == 1:
        regime = "unit" if m == 2 else "positive"
    elif q >= 4:
        regime = "positive"
    elif q == 3:
        regime = "unit" if m == 2 else "positive"
    else:
        regime = "zero" if m <= 4 else "unit" if m == 5 else "positive"
    return {"q": q, "m": m, "p": p, "f": f, "regime": regime, "exact_radius": p if m == 1 else None}
