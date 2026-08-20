
import './App.css'


import { useEffect, useState } from 'react';
import { WebSocket } from 'partysocket';

export function useJsonSocket(url) {
  const [data, setData] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.addEventListener('open', () => setReady(true));
    ws.addEventListener('close', () => setReady(false));
    ws.addEventListener('message', e => setData(JSON.parse(e.data)));
    return () => ws.close();
  }, [url]);

  return { data, ready };
}

function App() {
  console.log(RUW);
  const { lastJsonMessage, readyState } = useWebSocket('ws://localhost:8000/stream', {
    shouldReconnect: () => true,
    reconnectAttempts: Infinity,
    reconnectInterval: 1000,
  });

  return (
    <>
      <section id="center">
        <pre>{JSON.stringify(lastJsonMessage, null, 2)}</pre>
      </section>
    </>
  )
}

export default App
