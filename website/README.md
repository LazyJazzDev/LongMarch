# LongMarch project website

Static Chinese project documentation with four hash-linked tabs. No build step,
third-party JavaScript, or font service is required. With JavaScript disabled,
all sections remain readable.

Preview from the repository root:

```sh
python3 -m http.server 8000 --directory website
```

Visit http://localhost:8000. The `play/` links target the existing production
WebAssembly games and are not included in this local source-only preview.

The Pages workflow runs after website changes reach `main` (or manually on
`main`). It assembles the website with `play/` and `gameoflife/` from `gh-pages`,
then publishes via GitHub Pages Actions. The first deployment enables the
Actions publishing mode; repository administration permissions may be needed
if GitHub prevents that transition. Existing web applications remain sourced
from `gh-pages`. Do not delete that branch or those directories.

The four sections are `#home`, `#install`, `#architecture`, and `#examples`.
Keep commands consistent with CMake targets and platform support. Screenshot
URLs point to pinned commits in LongMarchAssetsLFS. The home illustration is
CSS concept art, not a renderer output.
