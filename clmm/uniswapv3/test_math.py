from decimal import Decimal, getcontext
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from uniswapv3.math import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    Q96,
    Q128,
    U256,
    compute_swap_step,
    div_up,
    fee_growth_delta_x128,
    get_amount0_delta,
    get_amount1_delta,
    get_next_sqrt_price_from_amount0_up,
    get_next_sqrt_price_from_amount1_down,
    get_next_sqrt_price_from_input,
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
    mul_div,
    mul_div_up,
    tokens_owed,
    wrap256,
)

getcontext().prec = 100

FEE_TIERS = [100, 500, 3000, 10000]

# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

ticks = st.integers(min_value=MIN_TICK, max_value=MAX_TICK - 1)
liquidities = st.integers(min_value=1, max_value=(1 << 128) - 1)
amounts = st.integers(min_value=0, max_value=1 << 120)
pos_amounts = st.integers(min_value=1, max_value=1 << 120)
jitters = st.integers(min_value=0, max_value=1 << 90)
fees = st.sampled_from(FEE_TIERS)


def sqrt_price_from(tick: int, jitter: int) -> int:
    """A valid sqrtPriceX96 at or slightly above a tick's ratio."""
    return min(get_sqrt_ratio_at_tick(tick) + jitter, MAX_SQRT_RATIO - 1)


# ---------------------------------------------------------------------------
# 1. handwritten vectors -- values asserted in v3-core TickMath.spec.ts
# ---------------------------------------------------------------------------


class TestTickMathVectors:
    def test_tick_zero_is_q96(self):
        assert get_sqrt_ratio_at_tick(0) == 79228162514264337593543950336
        assert get_sqrt_ratio_at_tick(0) == Q96

    def test_min_tick(self):
        assert get_sqrt_ratio_at_tick(MIN_TICK) == 4295128739

    def test_max_tick(self):
        assert (
            get_sqrt_ratio_at_tick(MAX_TICK)
            == 1461446703485210103287273052203988822378723970342
        )

    def test_out_of_range_tick_raises(self):
        with pytest.raises(ValueError):
            get_sqrt_ratio_at_tick(MAX_TICK + 1)
        with pytest.raises(ValueError):
            get_sqrt_ratio_at_tick(MIN_TICK - 1)

    def test_tick_at_min_ratio(self):
        assert get_tick_at_sqrt_ratio(MIN_SQRT_RATIO) == MIN_TICK

    def test_tick_at_ratio_just_below_max(self):
        assert get_tick_at_sqrt_ratio(MAX_SQRT_RATIO - 1) == MAX_TICK - 1

    def test_out_of_range_ratio_raises(self):
        with pytest.raises(ValueError):
            get_tick_at_sqrt_ratio(MIN_SQRT_RATIO - 1)
        with pytest.raises(ValueError):
            get_tick_at_sqrt_ratio(MAX_SQRT_RATIO)

    @pytest.mark.parametrize(
        "tick", [-887272, -500000, -60000, -1, 0, 1, 60000, 500000, 887272]
    )
    def test_against_decimal_oracle_within_hundredth_bip(self, tick):
        # v3-core TickMath.spec.ts: "is at most off by 1/100th of a bip"
        exact = (Decimal("1.0001") ** tick).sqrt() * (1 << 96)
        got = Decimal(get_sqrt_ratio_at_tick(tick))
        assert abs(got - exact) / exact <= Decimal("0.000001")


class TestMulDivVectors:
    def test_floor(self):
        assert mul_div(6, 7, 4) == 10

    def test_ceil(self):
        assert mul_div_up(6, 7, 4) == 11

    def test_exact_division_no_bump(self):
        assert mul_div(6, 8, 4) == 12
        assert mul_div_up(6, 8, 4) == 12

    def test_div_up(self):
        assert div_up(0, 5) == 0
        assert div_up(10, 5) == 2
        assert div_up(11, 5) == 3


class TestSwapStepVectors:
    def test_clean_amount1_in_at_price_one(self):
        # s = 2^96 (price 1.0), L = 2^96 -> 1 unit of amount1 moves s by
        # exactly 1. amt=1_000_000 @ 3000 pips: lessFee = 997_000.
        sp = Q96
        L = Q96
        target = get_sqrt_ratio_at_tick(60)
        sn, ain, aout, fee = compute_swap_step(sp, target, L, 1_000_000, 3000)
        assert sn == sp + 997_000
        assert ain == 997_000
        assert fee == 3_000
        assert ain + fee == 1_000_000
        real_out = Fraction(L << 96) * (sn - sp) / (sp * sn)
        assert aout == int(real_out)  # floor
        assert 0 <= real_out - aout < 1

    def test_seven_unit_dust_swap_floors_fee(self):
        # floor(7 * 0.9995) = 6 traded, 1 unit fee -- the integer behavior
        # a float engine cannot reproduce.
        sp = get_sqrt_ratio_at_tick(201840)
        L = 10**19
        target = get_sqrt_ratio_at_tick(201830)
        sn, ain, aout, fee = compute_swap_step(sp, target, L, 7, 500)
        assert ain == 6
        assert fee == 1
        assert sn < sp

    def test_full_dust_all_fee_no_trade(self):
        # amount too small to move price at huge liquidity: everything is
        # fee, price and output untouched.
        sp = Q96
        L = 1 << 127
        target = get_sqrt_ratio_at_tick(-60)
        sn, ain, aout, fee = compute_swap_step(sp, target, L, 1, 3000)
        assert sn == sp
        assert ain == 0
        assert aout == 0
        assert fee == 1

    def test_clamped_at_target_uses_grossed_up_fee(self):
        sp = get_sqrt_ratio_at_tick(0)
        target = get_sqrt_ratio_at_tick(10)
        L = 10**18
        amt = 1 << 100  # far more than the segment needs
        sn, ain, aout, fee = compute_swap_step(sp, target, L, amt, 3000)
        assert sn == target
        assert fee == mul_div_up(ain, 3000, 10**6 - 3000)
        assert ain + fee < amt


class TestFeeGrowthVectors:
    def test_flip_twice_is_identity(self):
        g = 12345 << 128
        fo = 777 << 128
        assert wrap256(g - wrap256(g - fo)) == fo

    def test_underflow_cancels(self):
        # feeGrowthOutside may wrap "negative"; deltas still come out right.
        fo = wrap256(5 - 10)
        inside_then = wrap256(100 - fo)
        inside_now = wrap256(150 - fo)
        assert fee_growth_delta_x128(inside_now, inside_then) == 50

    def test_tokens_owed(self):
        assert tokens_owed(3 << 128, 7) == 21
        assert tokens_owed((1 << 128) - 1, 1) == 0  # floor


# ---------------------------------------------------------------------------
# 2/3. property tests
# ---------------------------------------------------------------------------


class TestTickMathProperties:
    @given(t=ticks)
    def test_round_trip(self, t):
        assert get_tick_at_sqrt_ratio(get_sqrt_ratio_at_tick(t)) == t

    @given(t=ticks)
    def test_one_wei_below_boundary_is_previous_tick(self, t):
        # the boundary-landing case: exactly on a tick's ratio belongs to
        # that tick; one wei below belongs to the previous tick.
        assume(t > MIN_TICK)
        assert get_tick_at_sqrt_ratio(get_sqrt_ratio_at_tick(t) - 1) == t - 1

    @given(t=ticks, j=jitters)
    def test_defining_inequality(self, t, j):
        x = sqrt_price_from(t, j)
        tk = get_tick_at_sqrt_ratio(x)
        assert get_sqrt_ratio_at_tick(tk) <= x
        if tk + 1 <= MAX_TICK:
            assert x < get_sqrt_ratio_at_tick(tk + 1)

    @given(t1=ticks, t2=ticks)
    def test_strictly_monotonic(self, t1, t2):
        assume(t1 < t2)
        assert get_sqrt_ratio_at_tick(t1) < get_sqrt_ratio_at_tick(t2)


class TestMulDivProperties:
    @given(a=amounts, b=amounts, d=pos_amounts)
    def test_floor_bracketing(self, a, b, d):
        md = mul_div(a, b, d)
        assert md * d <= a * b < (md + 1) * d

    @given(a=amounts, b=amounts, d=pos_amounts)
    def test_up_equals_floor_plus_remainder_flag(self, a, b, d):
        assert mul_div_up(a, b, d) == mul_div(a, b, d) + (1 if (a * b) % d else 0)


class TestAmountDeltaProperties:
    @given(t1=ticks, t2=ticks, j1=jitters, j2=jitters, L=liquidities)
    def test_bracket_exact_rational(self, t1, t2, j1, j2, L):
        sa, sb = sorted((sqrt_price_from(t1, j1), sqrt_price_from(t2, j2)))
        assume(sa < sb)
        real0 = Fraction((L << 96) * (sb - sa), sb * sa)
        real1 = Fraction(L * (sb - sa), Q96)
        d0_dn = get_amount0_delta(sa, sb, L, False)
        d0_up = get_amount0_delta(sa, sb, L, True)
        d1_dn = get_amount1_delta(sa, sb, L, False)
        d1_up = get_amount1_delta(sa, sb, L, True)
        assert d0_dn <= real0 <= d0_up
        assert d1_dn <= real1 <= d1_up
        assert 0 <= d0_up - d0_dn <= 2  # double rounding in amount0
        assert 0 <= d1_up - d1_dn <= 1  # single rounding in amount1

    @given(t1=ticks, t2=ticks, L=liquidities, up=st.booleans())
    def test_symmetric_in_price_order(self, t1, t2, L, up):
        sa, sb = get_sqrt_ratio_at_tick(t1), get_sqrt_ratio_at_tick(t2)
        assert get_amount0_delta(sa, sb, L, up) == get_amount0_delta(sb, sa, L, up)
        assert get_amount1_delta(sa, sb, L, up) == get_amount1_delta(sb, sa, L, up)

    @given(t=ticks, L=liquidities, up=st.booleans())
    def test_zero_width_is_zero(self, t, L, up):
        s = get_sqrt_ratio_at_tick(t)
        assert get_amount0_delta(s, s, L, up) == 0
        assert get_amount1_delta(s, s, L, up) == 0


class TestNextPriceProperties:
    @given(t=ticks, j=jitters, L=liquidities, amt=amounts, zf1=st.booleans())
    def test_direction_and_zero_identity(self, t, j, L, amt, zf1):
        s = sqrt_price_from(t, j)
        if zf1:
            sn = get_next_sqrt_price_from_input(s, L, amt, True)
            assert sn <= s
        else:
            sn = get_next_sqrt_price_from_input(s, L, amt, False)
            assert sn >= s
        assert get_next_sqrt_price_from_input(s, L, 0, zf1) == s

    @given(t=ticks, j=jitters, L=liquidities, amt=pos_amounts)
    def test_never_undercharges_amount1_path(self, t, j, L, amt):
        # price rounded down on the way up -> the amount1 actually required
        # for the achieved move never exceeds what was paid.
        s = sqrt_price_from(t, j)
        sn = get_next_sqrt_price_from_amount1_down(s, L, amt, True)
        assert get_amount1_delta(s, sn, L, True) <= amt

    @given(t=ticks, j=jitters, L=liquidities, amt=pos_amounts)
    def test_never_undercharges_amount0_path(self, t, j, L, amt):
        # price rounded up on the way down -> conservative move; the
        # round-down amount0 for it never exceeds what was paid.
        s = sqrt_price_from(t, j)
        sn = get_next_sqrt_price_from_amount0_up(s, L, amt, True)
        assert get_amount0_delta(sn, s, L, False) <= amt


class TestSwapStepProperties:
    @given(
        t=ticks,
        j=jitters,
        L=liquidities,
        amt=pos_amounts,
        fee=fees,
        width=st.integers(min_value=1, max_value=2000),
        zf1=st.booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_core_invariants(self, t, j, L, amt, fee, width, zf1):
        assume(MIN_TICK < t - width and t + width < MAX_TICK)
        sp = sqrt_price_from(t, j)
        target = get_sqrt_ratio_at_tick(t - width if zf1 else t + width)
        assume((target <= sp) if zf1 else (target >= sp))

        sn, ain, aout, feeamt = compute_swap_step(sp, target, L, amt, fee)

        # price stays inside [target, start] (direction + clamp)
        lo, hi = (target, sp) if zf1 else (sp, target)
        assert lo <= sn <= hi

        # budget conservation: consumed fully unless clamped at the target
        if sn != target:
            assert ain + feeamt == amt
        else:
            assert ain + feeamt <= amt + 1

        # fee never below the nominal rate on the traded amount
        assert feeamt >= mul_div(ain, fee, 10**6)

        # pool never overpays: output <= exact rational output of the move
        if zf1:
            real_out = Fraction(L * (sp - sn), Q96)
        else:
            real_out = Fraction((L << 96) * (sn - sp), sp * sn)
        assert aout <= real_out

        # dust in, nothing out -> everything is fee (matches the chain)
        if aout == 0 and sn == sp:
            assert ain == 0 and feeamt == amt

    @given(t=ticks, j=jitters, L=liquidities, amt=pos_amounts, fee=fees)
    def test_degenerate_target_equals_start(self, t, j, L, amt, fee):
        # already at the target: zero trade, gross-up fee of zero input
        sp = sqrt_price_from(t, j)
        sn, ain, aout, feeamt = compute_swap_step(sp, sp, L, amt, fee)
        assert sn == sp and ain == 0 and aout == 0 and feeamt == 0


class TestFeeGrowthProperties:
    @given(
        g=st.integers(min_value=0, max_value=U256),
        fo=st.integers(min_value=0, max_value=U256),
    )
    def test_flip_twice_identity(self, g, fo):
        assert wrap256(g - wrap256(g - fo)) == wrap256(fo)

    @given(
        base=st.integers(min_value=0, max_value=U256),
        d1=st.integers(min_value=0, max_value=1 << 200),
        d2=st.integers(min_value=0, max_value=1 << 200),
    )
    def test_wrapped_deltas_recover_growth(self, base, d1, d2):
        # inside values may wrap arbitrarily; consecutive deltas still
        # recover the true growth d2 -- the property Position.update
        # depends on.
        then = wrap256(base + d1)
        now = wrap256(base + d1 + d2)
        assert fee_growth_delta_x128(now, then) == wrap256(d2)


# ---------------------------------------------------------------------------
# exact-output path (SwapMath.computeSwapStep with amount_remaining < 0)
# ---------------------------------------------------------------------------

from uniswapv3.math import get_next_sqrt_price_from_output  # noqa: E402

neg_amounts = st.integers(min_value=-(1 << 120), max_value=-1)


class TestSwapStepExactOutVectors:
    def test_delivers_exactly_when_not_clamped(self):
        sp = get_sqrt_ratio_at_tick(201840)
        L = 10**19
        target = get_sqrt_ratio_at_tick(201830)
        sn, ain, aout, fee = compute_swap_step(sp, target, L, -(10**9), 500)
        assert aout == 10**9
        assert sn != target and target < sn < sp
        assert fee == mul_div_up(ain, 500, 10**6 - 500)

    def test_clamped_at_target_delivers_segment_max(self):
        sp = get_sqrt_ratio_at_tick(0)
        target = get_sqrt_ratio_at_tick(-10)
        L = 10**18
        seg_out = get_amount1_delta(target, sp, L, False)
        sn, ain, aout, fee = compute_swap_step(sp, target, L, -(seg_out * 10), 3000)
        assert sn == target
        assert aout == seg_out
        assert fee == mul_div_up(ain, 3000, 10**6 - 3000)


class TestSwapStepExactOutProperties:
    @given(
        t=ticks,
        j=jitters,
        L=liquidities,
        amt=neg_amounts,
        fee=fees,
        width=st.integers(min_value=1, max_value=2000),
        zf1=st.booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_core_invariants(self, t, j, L, amt, fee, width, zf1):
        assume(MIN_TICK < t - width and t + width < MAX_TICK)
        sp = sqrt_price_from(t, j)
        target = get_sqrt_ratio_at_tick(t - width if zf1 else t + width)
        assume((target <= sp) if zf1 else (target >= sp))

        try:
            sn, ain, aout, feeamt = compute_swap_step(sp, target, L, amt, fee)
        except ValueError:
            assume(False)  # price under/overflow for extreme draws: out of scope

        lo, hi = (target, sp) if zf1 else (sp, target)
        assert lo <= sn <= hi  # direction + clamp

        assert aout <= -amt  # never overdeliver
        if sn != target:
            assert aout == -amt  # in full unless clamped

        assert feeamt == mul_div_up(ain, fee, 10**6 - fee)  # always gross-up

        # pool never undercharges: exact rational input for the move <= ain
        if zf1:
            real_in = Fraction((L << 96) * (sp - sn), sp * sn)
            real_out = Fraction(L * (sp - sn), Q96)
        else:
            real_in = Fraction(L * (sn - sp), Q96)
            real_out = Fraction((L << 96) * (sn - sp), sp * sn)
        assert ain >= real_in
        assert aout <= real_out  # never overpay

    @given(
        t=ticks,
        j=jitters,
        L=liquidities,
        amt=pos_amounts,
        fee=fees,
        width=st.integers(min_value=1, max_value=2000),
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_exact_out_of_exact_in_output_is_consistent(self, t, j, L, amt, fee, width):
        # zero-for-one: run exactIn, then ask exactOut for the output it
        # delivered; the output leg must be deliverable at a price no
        # further than the exactIn landing point.
        assume(MIN_TICK < t - width)
        sp = sqrt_price_from(t, j)
        target = get_sqrt_ratio_at_tick(t - width)
        assume(target <= sp)
        sn1, ain1, aout1, fee1 = compute_swap_step(sp, target, L, amt, fee)
        seg_out = get_amount1_delta(target, sp, L, False)
        # exclude the boundary case: if exactIn delivered the segment max,
        # exactOut legitimately clamps to the target (below sn1)
        assume(0 < aout1 < seg_out and sn1 != target)
        sn2, ain2, aout2, fee2 = compute_swap_step(sp, target, L, -aout1, fee)
        assert sn1 <= sn2 <= sp  # exactOut moves no further
        assert aout2 == aout1  # same delivery
        assert ain2 <= ain1 and fee2 <= fee1  # provable monotonicity
        assert ain2 + fee2 <= amt  # never costs more than paid


if __name__ == "__main__":
    unittest.main()
