from __future__ import annotations

from collections.abc import Iterable

def legendre_values(xi: float, modes: int) -> list[float]:

    if modes <= 0:
        raise ValueError("modes must be positive")
    values = [1.0]
    if modes == 1:
        return values

    values.append(float(xi))
    for n in range(1, modes - 1):
        pn = values[n]
        pnm1 = values[n - 1]
        values.append(((2 * n + 1) * xi * pn - n * pnm1) / (n + 1))
    return values

def legendre_matrix(points: Iterable[float], modes: int) -> list[list[float]]:
    return [legendre_values(float(xi), modes) for xi in points]
