import { useStream } from './lib';

const WS_URL = import.meta.env.VITE_BACKEND_URL ?? 'ws://localhost:8080';

function fmt(n: number | null, w: number): string {
  if (n === null || !Number.isFinite(n)) {
    return '';
  }
  return n.toLocaleString(undefined, { minimumFractionDigits: w, maximumFractionDigits: w });
}

function App() {
  const { latest, open, closed, ready } = useStream(WS_URL);

  const quotes = Object.entries(latest)
    .map(([pair, q]) => ({ pair, q, basis: q.basis - 2 * q.cost }))
    .toSorted((a, b) => b.basis - a.basis);

  const openRows = Object.entries(open)
    .map(([pair, o]) => {
      const entry = o.basis - 2 * o.cost;
      const l = latest[pair];
      const basis = l ? l.basis - 2 * l.cost : null;
      const pnl = basis === null ? null : entry - basis;
      return { pair, entry, basis, pnl };
    })
    .toSorted((a, b) => (b.pnl ?? -Infinity) - (a.pnl ?? -Infinity));
  const openPnl = openRows.reduce((sum, r) => sum + (r.pnl ?? 0), 0);

  const closedRows = closed
    .map(c => {
      const open = c.entry_basis - 2 * c.entry_cost;
      const close = c.exit_basis - 2 * c.exit_cost;
      const pnl = open - close;
      return { pair: c.pair, open, close, pnl, exit_at: c.exit_at };
    })
    .toSorted((a, b) => b.exit_at - a.exit_at);
  const closedPnl = closedRows.reduce((sum, r) => sum + r.pnl, 0);

  return (
    <>
      <section id="positions" className="mx-auto max-w-4xl px-4 py-8">
        {ready && (openRows.length === 0 ? (
          <p className="px-4 py-2 text-center text-sm text-gray-500">No open positions</p>
        ) : (
          <table className="w-full border-collapse rounded-lg border border-gray-200 shadow-sm">
            <tr className="bg-gray-50">
              <th className="px-4 py-3 text-left text-sm font-semibold">ticker</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">pnl (net)</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">basis (net)</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">entry (net)</th>
            </tr>
            {openRows.map(r => (
              <tr key={r.pair}>
                <td className='px-4 py-2 text-left'>{r.pair}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(r.pnl, 1)}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(r.basis, 1)}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(r.entry, 1)}</td>
              </tr>
            ))}
            <tr className="bg-gray-50 font-semibold">
              <td className='px-4 py-3 text-left'>total</td>
              <td className='px-4 py-3 text-right tabular-nums'>{fmt(openPnl, 1)}</td>
              <td className='px-4 py-3 text-right' colSpan={2}></td>
            </tr>
          </table>
        ))}
      </section>
      <section id="closed" className="mx-auto max-w-4xl px-4 py-8">
        {ready && (closedRows.length === 0 ? (
          <p className="px-4 py-2 text-center text-sm text-gray-500">No closed positions</p>
        ) : (
          <table className="w-full border-collapse rounded-lg border border-gray-200 shadow-sm">
            <tr className="bg-gray-50">
              <th className="px-4 py-3 text-left text-sm font-semibold">ticker</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">open (net)</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">close (net)</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">pnl (net)</th>
            </tr>
            {closedRows.map(r => (
              <tr key={`${r.pair}-${r.exit_at}`}>
                <td className='px-4 py-2 text-left'>{r.pair}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(r.open, 1)}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(r.close, 1)}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(r.pnl, 1)}</td>
              </tr>
            ))}
            <tr className="bg-gray-50 font-semibold">
              <td className='px-4 py-3 text-left'>total</td>
              <td className='px-4 py-3 text-right' colSpan={2}></td>
              <td className='px-4 py-3 text-right tabular-nums'>{fmt(closedPnl, 1)}</td>
            </tr>
          </table>
        ))}
      </section>
      <section id="quotes" className="mx-auto max-w-4xl px-4 py-8">
        {ready && (
          <table className="w-full border-collapse rounded-lg border border-gray-200 shadow-sm">
            <tr className="bg-gray-50">
              <th className="px-4 py-3 text-left text-sm font-semibold">ticker</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">basis (net)</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">spot mid (spread bps)</th>
              <th className="px-4 py-3 text-right text-sm font-semibold">future mid (spread bps)</th>
            </tr>
            {quotes.map(({ pair, q, basis }) => (
              <tr key={pair}>
                <td className='px-4 py-2 text-left'>{pair}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(basis, 1)}</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(q.spot_mid, 5)} ({fmt(q.spot_spread, 1)})</td>
                <td className='px-4 py-2 text-right tabular-nums'>{fmt(q.fut_mid, 5)} ({fmt(q.fut_spread, 1)})</td>
              </tr>
            ))}
          </table>
        )}
      </section>
    </>
  );
}

export default App;
