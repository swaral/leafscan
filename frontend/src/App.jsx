import { useEffect, useState } from 'react';
import { BarChart3, BookOpen, History as HistoryIcon, Layers, Leaf, Moon, ScanSearch, Sun } from 'lucide-react';
import { api } from './api';
import Batch from './pages/Batch';
import Diagnose from './pages/Diagnose';
import History from './pages/History';
import Library from './pages/Library';
import ModelPage from './pages/ModelPage';

const ROUTES = {
  diagnose: { label: 'Diagnose', Icon: ScanSearch, Page: Diagnose },
  batch: { label: 'Batch', Icon: Layers, Page: Batch },
  history: { label: 'History', Icon: HistoryIcon, Page: History },
  model: { label: 'Model', Icon: BarChart3, Page: ModelPage },
  library: { label: 'Library', Icon: BookOpen, Page: Library },
};

const currentRoute = () => {
  const r = window.location.hash.replace('#/', '');
  return ROUTES[r] ? r : 'diagnose';
};

function readTheme() {
  try { return localStorage.getItem('theme'); } catch { return null; }
}

export default function App() {
  const [route, setRoute] = useState(currentRoute);
  const [theme, setTheme] = useState(readTheme);
  const [apiState, setApiState] = useState('checking');

  useEffect(() => {
    const onHash = () => { setRoute(currentRoute()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try { theme ? localStorage.setItem('theme', theme) : localStorage.removeItem('theme'); } catch { /* storage blocked */ }
  }, [theme]);

  useEffect(() => {
    const check = () => api.health().then((h) => setApiState(h.model_loaded ? 'ok' : 'down')).catch(() => setApiState('down'));
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  const isDark = theme ? theme === 'dark' : window.matchMedia?.('(prefers-color-scheme: dark)').matches;
  const { Page } = ROUTES[route];

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <a href="#/diagnose" className="logo">
            <span className="logo-mark"><Leaf size={18} /></span> LeafScan
          </a>
          <nav className="nav">
            {Object.entries(ROUTES).map(([k, { label, Icon }]) => (
              <a key={k} href={`#/${k}`} className={route === k ? 'active' : ''} aria-label={label}>
                <Icon size={16} /> <span>{label}</span>
              </a>
            ))}
          </nav>
          <div className="topbar-right">
            <span className={`api-dot ${apiState}`} title="API status">
              <i /> {apiState === 'ok' ? 'Model online' : apiState === 'down' ? 'API offline' : 'Connecting'}
            </span>
            <button className="btn ghost icon-btn" onClick={() => setTheme(isDark ? 'light' : 'dark')} aria-label="Toggle dark mode">
              {isDark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </header>
      <main><Page /></main>
      <footer>
        LeafScan · CNN + FastAPI + React · for guidance, not a substitute for expert diagnosis
      </footer>
    </>
  );
}
