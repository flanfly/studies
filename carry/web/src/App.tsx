import { useEffect, useState } from 'react';
import { WebSocket } from 'partysocket';

type PairState = {
  funding: number | null;
  spot_ask_price: number | null;
  spot_ask_qty: number | null;
  spot_bid_price: number | null;
  spot_bid_qty: number | null;
  future_ask_price: number | null;
  future_ask_qty: number | null;
  future_bid_price: number | null;
  future_bid_qty: number | null;
  last_future: number | null;
  last_spot: number | null;
  mark: number | null;
  index: number | null;
  spot_mid: number | null;
  spot_spread_bps: number | null;
  future_mid: number | null;
  future_spread_bps: number | null;
  basis_bps: number | null;
};
const PairStateFieldWidths = {
  'basis_bps': 1, 'funding': 1, 'spot_mid': 5, 'spot_spread_bps': 1, 'future_mid': 5, 'future_spread_bps': 1
} as const satisfies Record<string, number>

type Row = Record<string, string | number | null>

export function useWebSocket(url: string) {
  const [data, setData] = useState<[string, Row][]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.addEventListener('open', () => setReady(true));
    ws.addEventListener('close', () => setReady(false));
    ws.addEventListener('message', e => {
      const msg = JSON.parse(e.data);
      if (typeof msg !== 'object' || msg === null) {
        return
      }

      const pairs = Object.entries(msg as Record<string, PairState>)
        .toSorted((a, b) => (b[1].basis_bps ?? 0) - (a[1].basis_bps ?? 0))

      const fmt: [string, Row][] = pairs.map(([name, state]) => {
        const row: Row = { ...state }
        for (const [fn, w] of Object.entries(PairStateFieldWidths)) {
          const val = state[fn as keyof PairState]
          if (typeof val === 'number') {
            row[fn] = val.toLocaleString(undefined, { minimumFractionDigits: w, maximumFractionDigits: w })
          }
        }
        return [name, row]
      })

      setData(fmt)
    })
    return () => ws.close();
  }, [url]);

  return { data, ready };
}

function App() {
  const { data, ready } = useWebSocket('ws://ether:8000/ws');

  return (
    <section id="center" className="mx-auto max-w-4xl px-4 py-8">
      {ready && data !== null && (
        <table className="w-full border-collapse rounded-lg border border-gray-200 shadow-sm">
          <tr className="bg-gray-50">
            <th className="px-4 py-3 text-left text-sm font-semibold">ticker</th>
            <th className="px-4 py-3 text-right text-sm font-semibold">basis</th>
            <th className="px-4 py-3 text-right text-sm font-semibold">funding</th>
            <th className="px-4 py-3 text-right text-sm font-semibold">spot mid (spread bps)</th>
            <th className="px-4 py-3 text-right text-sm font-semibold">future mid (spread bps)</th>
          </tr>
          {data.map(p => (
            <tr key={p[0]}>
              <td className='px-4 py-2 text-left'>{p[0]}</td>
              <td className='px-4 py-2 text-right tabular-nums'>{p[1]['basis_bps']}</td>
              <td className='px-4 py-2 text-right tabular-nums'>{p[1]['funding']}</td>
              <td className='px-4 py-2 text-right tabular-nums'>{p[1]['spot_mid']} ({p[1]['spot_spread_bps']})</td>
              <td className='px-4 py-2 text-right tabular-nums'>{p[1]['future_mid']} ({p[1]['future_spread_bps']})</td>
            </tr>))}
        </table>
      )}
    </section>
  )
}

export default App
