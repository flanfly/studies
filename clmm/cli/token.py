from dataclasses import dataclass
import re
import numpy as np

from functools import cached_property

from uniswapv3.load import Pool
import uniswapv3.math as v3math

TOKEN_RE = r"([0-9.]+)\s*([a-zA-Z]+)"


@dataclass(frozen=True)
class Token:
    symbol: str
    ord: int
    amount: int
    decimals: int

    @cached_property
    def str(self) -> str:
        return f"{self.amount / pow(10, self.decimals)}{self.symbol}"


def parse_amount(s: str, meta: Pool) -> Token:
    import re

    m = re.match(TOKEN_RE, s)
    if not m:
        raise ValueError(f"Could not parse amount: {s}")

    sym = m.group(2)
    amount = float(m.group(1))

    print(amount, sym)

    if sym.lower() == meta.symbol0.lower():
        return Token(
            symbol=sym, ord=0, amount=int(amount * pow(10, meta.d0)), decimals=meta.d0
        )
    elif sym.lower() == meta.symbol1.lower():
        return Token(
            symbol=sym, ord=1, amount=int(amount * pow(10, meta.d1)), decimals=meta.d1
        )
    else:
        raise ValueError(f"Unknown symbol: {sym}")


def token_to_sqrtp(tok: Token, meta: Pool) -> int:
    match tok.ord:
        case 0:
            ratio = 10**meta.d1 / tok.amount
        case 1:
            ratio = tok.amount / 10**meta.d0
        case _:
            assert False

    return int(np.sqrt(ratio) * (2**96))


def liquidity_for_value(
    sqrt_price_x96: int, tick_lower: int, tick_upper: int, tok: Token
) -> int:
    """L such that the position's total value at the current price equals tok.amount."""
    sa = v3math.get_sqrt_ratio_at_tick(tick_lower)
    sb = v3math.get_sqrt_ratio_at_tick(tick_upper)
    sc = min(max(sqrt_price_x96, sa), sb)  # clamp: one-sided ranges
    a0 = v3math.get_amount0_delta(sc, sb, v3math.Q96, False)  # token0 leg per Q of L
    a1 = v3math.get_amount1_delta(sa, sc, v3math.Q96, False)  # token1 leg per Q of L
    a1_in_0 = a1 * (1 << 192) // (sqrt_price_x96**2)  # token1 leg valued in token0
    if tok.ord == 0:
        return tok.amount * v3math.Q96 // (a0 + a1_in_0)
    a0_in_1 = a0 * (sqrt_price_x96**2) // (1 << 192)
    return tok.amount * v3math.Q96 // (a0_in_1 + a1)
