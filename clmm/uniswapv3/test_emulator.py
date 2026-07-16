import unittest
from pathlib import Path

import polars as pl
import numpy as np

from uniswapv3.emulator import Emulator
from uniswapv3.load import from_ethereum
import uniswapv3.math as v3math

import logging as l
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")


class TestEmulator(unittest.IsolatedAsyncioTestCase):

    def assertClose(self, a, b, rtol=np.finfo(np.float32).eps):
        self.assertLessEqual(
            abs(a - b), max(abs(a), abs(b), 1) * rtol, f"{a} not close to {b}"
        )

    async def asyncSetUp(self):
        from os import getenv

        testdir = Path(__file__).resolve().parent.parent / "uniswap-v3-usdc-eth"

        df = (
            pl.read_parquet(f"{testdir}/ethereum__logs__*.parquet")
            .join(
                pl.read_parquet(f"{testdir}/ethereum__blocks__*.parquet").select(
                    pl.col("block_number"),
                    ts=pl.from_epoch(
                        pl.col("timestamp"), time_unit="s"
                    ).dt.replace_time_zone("UTC"),
                ),
                on=["block_number"],
            )
            .sort(["block_number", "transaction_index", "log_index"])
        )
        swaps, liq, params, meta = await from_ethereum(df)
        self.swaps = swaps
        self.liq = liq
        self.meta = meta
        self.contract = Emulator(
            meta.sqrt_price_x96,
            meta.tick,
            meta.liquidity,
            meta.ticks,
            meta.tick_spacing,
            meta.fee_pips,
            meta.protocol_fraction,
            meta.max_liquidity_per_tick,
        )

    async def test_swap(self):
        from tqdm import tqdm

        blocks = set(self.swaps["block_number"].to_list()) | set(
            self.liq["block_number"].to_list()
        )

        for bn in tqdm(sorted(blocks)):
            swaps = self.swaps.filter(pl.col("block_number") == bn)
            liq = self.liq.filter(pl.col("block_number") == bn)

            for ord in sorted(set(swaps["ord"].to_list()) | set(liq["ord"].to_list())):
                for row in swaps.filter(pl.col("ord") == ord).iter_rows(named=True):
                    if row["amount0"] > 0:
                        token_in = 0
                        amount_in = row["amount0"]
                        amount_out = row["amount1"]
                    else:
                        token_in = 1
                        amount_in = row["amount1"]
                        amount_out = row["amount0"]

                    # assume exact_input
                    est = self.contract.store_swap_state()
                    swap0, swap1, remaining = self.contract.swap(
                        bn, token_in, int(amount_in)
                    )

                    if (
                        self.contract.tick != row["tick"]
                        or self.contract.sqrt_price_x96 != row["sqrt_price_x96"]
                        or remaining != 0
                    ):
                        # try exact_output
                        self.contract.load_swap_state(est)
                        swap0, swap1, remaining = self.contract.swap(
                            bn, token_in, int(amount_out)
                        )

                        if (
                            self.contract.tick != row["tick"]
                            or self.contract.sqrt_price_x96 != row["sqrt_price_x96"]
                            or remaining != 0
                        ):
                            self.contract.load_swap_state(est)
                            swap0, swap1, remaining = self.contract.swap(
                                bn,
                                token_in,
                                int(amount_in),
                                limit_sqrt_x96=int(row["sqrt_price_x96"]),
                            )

                    self.assertEqual(self.contract.tick, row["tick"])
                    self.assertEqual(
                        self.contract.sqrt_price_x96, row["sqrt_price_x96"]
                    )
                    self.assertEqual(self.contract.liquidity, row["liquidity"])
                    self.assertEqual(remaining, 0)
                    self.assertEqual(swap0, row["amount0"])
                    self.assertEqual(swap1, row["amount1"])

                for row in liq.filter(pl.col("ord") == ord).iter_rows(named=True):
                    amount0, amount1 = self.contract.modify_liquidity(
                        row["tick_lower"],
                        row["tick_upper"],
                        row["liquidity"],
                    )
                    self.assertEqual(
                        int(np.sign(row["liquidity"])) * row["amount0"], amount0
                    )
                    self.assertEqual(
                        int(np.sign(row["liquidity"])) * row["amount1"], amount1
                    )


if __name__ == "__main__":
    unittest.main()
