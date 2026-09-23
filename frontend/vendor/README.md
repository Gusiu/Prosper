# Third-party files served by the dashboard

The page used to load these from jsDelivr and Google Fonts at runtime. That made
the dashboard depend on the network for its charts, and `/npm/chart.js` carried
no version at all, so whatever Chart.js was published next was what the page ran.
They are pinned copies now, served by the same app as everything else.

| File | Source | SHA-256 (prefix) |
| --- | --- | --- |
| `chart.umd.min.js` | https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js | `48444a82d4edcb5b` |
| `lightweight-charts.standalone.production.js` | https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js | `c7dda807d662a95b` |
| `fonts/inter-latin.woff2` | Google Fonts, Inter (variable), latin | `3100e775e8616cd2` |
| `fonts/inter-latin-ext.woff2` | Google Fonts, Inter (variable), latin-ext | `34b9c504cab7a73e` |
| `fonts/fira-code-latin.woff2` | Google Fonts, Fira Code (variable), latin | `771bf4b79a97fc00` |
| `fonts/fira-code-latin-ext.woff2` | Google Fonts, Fira Code (variable), latin-ext | `801677342d1191c5` |

| Library | Version | License |
| --- | --- | --- |
| Chart.js | 4.5.1 | MIT - `LICENSE-chart.js.md` |
| TradingView Lightweight Charts | 4.2.3 | Apache 2.0 - `LICENSE-lightweight-charts` |
| Inter | variable | SIL OFL 1.1 - `fonts/OFL-Inter.txt` |
| Fira Code | variable | SIL OFL 1.1 - `fonts/OFL-FiraCode.txt` |

To upgrade a library, replace its file, update this table, and check the
Analysis and Evaluation tabs: both draw with it.
