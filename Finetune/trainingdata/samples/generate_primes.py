from math import isqrt
from pathlib import Path


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = isqrt(n)
    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False
    return True


def first_n_primes(n: int) -> list[int]:
    primes = []
    candidate = 2
    while len(primes) < n:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return primes


def main() -> None:
    out_path = Path(__file__).resolve().parent / "math_primes.txt"
    primes = first_n_primes(10000)
    out_path.write_text("\n".join(str(p) for p in primes), encoding="utf-8")
    print(f"Wrote {len(primes)} primes to {out_path}")


if __name__ == "__main__":
    main()
