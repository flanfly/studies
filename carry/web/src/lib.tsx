import { useEffect, useState } from 'react';
import { WebSocket } from 'partysocket';

// wire shape of the proto `Basis`, keyed by pair in `latest`
export type Basis = {
  is_aggregation: boolean;
  observed_at: number;
  pair: string;
  basis: number;
  cost: number;
  spot_bid: number;
  spot_ask: number;
  spot_mid: number;
  spot_spread: number;
  fut_bid: number;
  fut_ask: number;
  fut_mid: number;
  fut_spread: number;
};

// wire shape of the proto `Open`, keyed by pair in `open`
export type Open = {
  pair: string;
  basis: number;
  cost: number;
  deviation: number;
};

// wire shape of the proto `Close`, keyed by pair in `closed`
export type Close = {
  pair: string;
  entry_at: number;
  entry_basis: number;
  entry_deviation: number;
  entry_cost: number;
  exit_at: number;
  exit_basis: number;
  exit_cost: number;
  exit_deviation: number;
};

export type Stream = {
  latest: Record<string, Basis>;
  open: Record<string, Open>;
  closed: Close[];
};

export function useStream(url: string) {
  const [stream, setStream] = useState<Stream>({ latest: {}, open: {}, closed: [] });
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(`${url}/ws`);
    ws.addEventListener('open', () => setReady(true));
    ws.addEventListener('close', () => setReady(false));
    ws.addEventListener('message', e => {
      const msg = JSON.parse(e.data);
      if (typeof msg !== 'object' || msg === null) {
        return;
      }
      setStream({
        latest: msg.latest ?? {},
        open: msg.open ?? {},
        closed: msg.closed ?? [],
      });
    });
    return () => ws.close();
  }, [url]);

  return { ...stream, ready };
}
