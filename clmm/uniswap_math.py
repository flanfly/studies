"""Exact integer port of Uniswap v3 core math.

Bit-for-bit faithful to v3-core Solidity for all realistic inputs:
  - TickMath.getSqrtRatioAtTick / getTickAtSqrtRatio
  - SqrtPriceMath.getNextSqrtPriceFromInput, getAmount0Delta, getAmount1Delta
  - SwapMath.computeSwapStep (exact-input path)
  - Q128 fee-growth arithmetic with uint256 wraparound

All prices are Q64.96 ints (sqrtPriceX96). All amounts are raw-unit ints.
Python ints are arbitrary precision, so FullMath's 512-bit machinery is
unnecessary: mulDiv(a, b, d) == a * b // d exactly.
"""

import math

Q32 = 1 << 32
Q96 = 1 << 96
Q128 = 1 << 128
U256 = (1 << 256) - 1
MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

_TICK_C = [
    0xFFFCB933BD6FAD37AA2D162D1A594001,
    0xFFF97272373D413259A46990580E213A,
    0xFFF2E50F5F656932EF12357CF3C7FDCC,
    0xFFE5CACA7E10E4E61C3624EAA0941CD0,
    0xFFCB9843D60F6159C9DB58835C926644,
    0xFF973B41FA98C081472E6896DFB254C0,
    0xFF2EA16466C96A3843EC78B326B52861,
    0xFE5DEE046A99A2A811C461F1969C3053,
    0xFCBE86C7900A88AEDCFFC83B479AA3A4,
    0xF987A7253AC413176F2B074CF7815E54,
    0xF3392B0822B70005940C7A398E4B70F3,
    0xE7159475A2C29B7443B29C7FA6E889D9,
    0xD097F3BDFD2022B8845AD8F792AA5825,
    0xA9F746462D870FDF8A65DC1F90E061E5,
    0x70D869A156D2A1B890BB3DF62BAF32F7,
    0x31BE135F97D08FD981231505542FCFA6,
    0x9AA508B5B7A84E1C677DE54F3E99BC9,
    0x5D6AF8DEDB81196699C329225EE604,
    0x2216E584F5FA1EA926041BEDFE98,
    0x48A170391F7DC42444E8FA2,
]

_LOG_1_0001 = math.log(1.0001)


def mul_div(a: int, b: int, d: int) -> int:
    """FullMath.mulDiv -- floor(a * b / d)."""
    return a * b // d


def mul_div_up(a: int, b: int, d: int) -> int:
    """FullMath.mulDivRoundingUp -- ceil(a * b / d)."""
    return -(a * b // -d)


def div_up(a: int, d: int) -> int:
    """UnsafeMath.divRoundingUp -- ceil(a / d)."""
    return -(a // -d)


def get_sqrt_ratio_at_tick(tick: int) -> int:
    """TickMath.getSqrtRatioAtTick -- exact. Returns Q64.96."""
    a = abs(tick)
    if a > MAX_TICK:
        raise ValueError("tick out of range")
    ratio = _TICK_C[0] if a & 1 else (1 << 128)
    for i in range(1, 20):
        if a & (1 << i):
            ratio = (ratio * _TICK_C[i]) >> 128
    if tick > 0:
        ratio = U256 // ratio
    return (ratio >> 32) + (1 if ratio & (Q32 - 1) else 0)


def get_tick_at_sqrt_ratio(sqrt_price_x96: int) -> int:
    """TickMath.getTickAtSqrtRatio -- exact.

    Spec: greatest tick such that getSqrtRatioAtTick(tick) <= sqrt_price_x96.
    Float log gives an estimate; integer comparisons against the exact
    ratios make the result exact.
    """
    if not (MIN_SQRT_RATIO <= sqrt_price_x96 < MAX_SQRT_RATIO):
        raise ValueError("sqrt ratio out of range")
    t = int(math.floor(2.0 * math.log(sqrt_price_x96 / Q96) / _LOG_1_0001))
    t = max(MIN_TICK, min(MAX_TICK - 1, t))
    while get_sqrt_ratio_at_tick(t) > sqrt_price_x96:
        t -= 1
    while t < MAX_TICK - 1 and get_sqrt_ratio_at_tick(t + 1) <= sqrt_price_x96:
        t += 1
    return t


def get_next_sqrt_price_from_amount0_up(
    sqrt_px96: int, liquidity: int, amount: int, add: bool
) -> int:
    """SqrtPriceMath.getNextSqrtPriceFromAmount0RoundingUp.

    Python note: Solidity has an overflow-fallback branch reachable only
    when amount * sqrtPX96 >= 2**256 (amounts beyond ~2**146 at mainnet
    prices); unreachable for real pools, so only the primary branch is
    ported.
    """
    if amount == 0:
        return sqrt_px96
    numerator1 = liquidity << 96
    product = amount * sqrt_px96
    if add:
        return mul_div_up(numerator1, sqrt_px96, numerator1 + product)
    if numerator1 <= product:
        raise ValueError("price underflow")
    return mul_div_up(numerator1, sqrt_px96, numerator1 - product)


def get_next_sqrt_price_from_amount1_down(
    sqrt_px96: int, liquidity: int, amount: int, add: bool
) -> int:
    """SqrtPriceMath.getNextSqrtPriceFromAmount1RoundingDown."""
    if add:
        return sqrt_px96 + (amount << 96) // liquidity
    quotient = div_up(amount << 96, liquidity)
    if sqrt_px96 <= quotient:
        raise ValueError("price underflow")
    return sqrt_px96 - quotient


def get_next_sqrt_price_from_input(
    sqrt_px96: int, liquidity: int, amount_in: int, zero_for_one: bool
) -> int:
    """SqrtPriceMath.getNextSqrtPriceFromInput."""
    if zero_for_one:
        return get_next_sqrt_price_from_amount0_up(
            sqrt_px96, liquidity, amount_in, True
        )
    return get_next_sqrt_price_from_amount1_down(sqrt_px96, liquidity, amount_in, True)


def get_amount0_delta(
    sqrt_a_x96: int, sqrt_b_x96: int, liquidity: int, round_up: bool
) -> int:
    """SqrtPriceMath.getAmount0Delta -- L * (sb - sa) / (sa * sb), Q96-scaled."""
    if sqrt_a_x96 > sqrt_b_x96:
        sqrt_a_x96, sqrt_b_x96 = sqrt_b_x96, sqrt_a_x96
    numerator1 = liquidity << 96
    numerator2 = sqrt_b_x96 - sqrt_a_x96
    if round_up:
        return div_up(mul_div_up(numerator1, numerator2, sqrt_b_x96), sqrt_a_x96)
    return mul_div(numerator1, numerator2, sqrt_b_x96) // sqrt_a_x96


def get_amount1_delta(
    sqrt_a_x96: int, sqrt_b_x96: int, liquidity: int, round_up: bool
) -> int:
    """SqrtPriceMath.getAmount1Delta -- L * (sb - sa), Q96-scaled."""
    if sqrt_a_x96 > sqrt_b_x96:
        sqrt_a_x96, sqrt_b_x96 = sqrt_b_x96, sqrt_a_x96
    if round_up:
        return mul_div_up(liquidity, sqrt_b_x96 - sqrt_a_x96, Q96)
    return mul_div(liquidity, sqrt_b_x96 - sqrt_a_x96, Q96)


def compute_swap_step(
    sqrt_price_x96: int,
    sqrt_price_target_x96: int,
    liquidity: int,
    amount_remaining: int,
    fee_pips: int,
):
    """SwapMath.computeSwapStep, exact-input path (amount_remaining > 0).

    Returns (sqrt_price_next_x96, amount_in, amount_out, fee_amount).
    """
    zero_for_one = sqrt_price_x96 >= sqrt_price_target_x96
    amount_remaining_less_fee = mul_div(amount_remaining, 10**6 - fee_pips, 10**6)

    if zero_for_one:
        amount_in = get_amount0_delta(
            sqrt_price_target_x96, sqrt_price_x96, liquidity, True
        )
    else:
        amount_in = get_amount1_delta(
            sqrt_price_x96, sqrt_price_target_x96, liquidity, True
        )

    if amount_remaining_less_fee >= amount_in:
        sqrt_price_next_x96 = sqrt_price_target_x96
    else:
        sqrt_price_next_x96 = get_next_sqrt_price_from_input(
            sqrt_price_x96, liquidity, amount_remaining_less_fee, zero_for_one
        )

    is_max = sqrt_price_target_x96 == sqrt_price_next_x96

    if zero_for_one:
        if not is_max:
            amount_in = get_amount0_delta(
                sqrt_price_next_x96, sqrt_price_x96, liquidity, True
            )
        amount_out = get_amount1_delta(
            sqrt_price_next_x96, sqrt_price_x96, liquidity, False
        )
    else:
        if not is_max:
            amount_in = get_amount1_delta(
                sqrt_price_x96, sqrt_price_next_x96, liquidity, True
            )
        amount_out = get_amount0_delta(
            sqrt_price_x96, sqrt_price_next_x96, liquidity, False
        )

    if not is_max:
        fee_amount = amount_remaining - amount_in
    else:
        fee_amount = mul_div_up(amount_in, fee_pips, 10**6 - fee_pips)

    return sqrt_price_next_x96, amount_in, amount_out, fee_amount


def wrap256(x: int) -> int:
    """uint256 wraparound -- Solidity's unchecked add/sub semantics.

    Fee-growth values (X128) rely on modular arithmetic: feeGrowthOutside
    and feeGrowthInside deltas may 'underflow' on-chain and cancel out
    later. Apply after every fee-growth subtraction/addition.
    """
    return x & U256


def fee_growth_delta_x128(inside_now: int, inside_last: int) -> int:
    """Position.update's (feeGrowthInside - feeGrowthInsideLast), wrapped."""
    return wrap256(inside_now - inside_last)


def tokens_owed(fee_growth_delta: int, liquidity: int) -> int:
    """FullMath.mulDiv(delta, liquidity, Q128)."""
    return mul_div(fee_growth_delta, liquidity, Q128)


if __name__ == "__main__":
    assert get_sqrt_ratio_at_tick(0) == Q96
    assert get_sqrt_ratio_at_tick(MIN_TICK) == MIN_SQRT_RATIO
    assert get_sqrt_ratio_at_tick(MAX_TICK) == MAX_SQRT_RATIO
    print("self-test ok")
