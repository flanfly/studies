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
  'basis_bps': 1,'funding': 1,'spot_mid': 5,'spot_spread_bps': 1,'future_mid': 5,'future_spread_bps': 1}


export function useWebSocket(url: string) {
  const [data, setData] = useState<[string, Record<string, string>]>([]);
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

      const pairs = Object.entries(msg).toSorted((a,b) => b[1].basis_bps - a[1].basis_bps)
      const fmt = pairs.map((p: [string, PairState]) => {
        for (const fn of Object.keys(PairStateFieldWidths)) {
          const w = PairStateFieldWidths[fn]
          const val = p[1][fn]
          if (val !== null && val !== undefined) {
            p[1][fn] = val.toLocaleString(undefined, { minimumFractionDigits: w })
          }
        }

        return p
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
    <section id="center">
      {ready && data !== null && (
        <table>
        <tr>
          <th>ticker</th>
          <th>basis</th>
          <th>funding</th>
          <th>spot mid (spread bps)</th>
          <th>future mid (spread bps)</th>
        </tr>
        {data.map(p => (
          <tr key={p[0]}>
            <td className='p-2 text-left'>{p[0]}</td>
            <td className='p-2 text-right tabular-nums'>{p[1]['basis_bps']}</td>
            <td className='p-2 text-right tabular-nums'>{p[1]['funding']}</td>
            <td className='p-2 text-right tabular-nums'>{p[1]['spot_mid']} ({p[1]['spot_spread_bps']})</td>
            <td className='p-2 text-right tabular-nums'>{p[1]['future_mid']} ({p[1]['future_spread_bps']})</td>
          </tr>))}
        </table>
      )}
    </section>
  )
}

export default App
