# Where these came from

Rendered from the talk deck's standalone figure sources in the
`sensys27-observability` repository, so that the README and the paper cannot
drift apart:

| here | source |
|---|---|
| `onemodality.png` | `figures/onemodality-slide.tex` |
| `sigma-three.png` | `figures/sigma-three-slide.tex` |
| `collect.png` | `figures/collect-slide.tex` |
| `contract.png` | `figures/contract-slide.tex` |

To regenerate after the deck changes:

```bash
pdftoppm -png -r 200 -singlefile <repo>/figures/<name>-slide.pdf docs/img/<name>
```

PNG rather than PDF because GitHub does not render PDFs inline. Keep them at
200 dpi: the figures are 13.4 cm wide by design, and lower resolutions make the
axis labels unreadable on the rendered page.
