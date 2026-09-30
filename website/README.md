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

The Pages workflow runs after website changes reach `main`, after published
snapshot or web application changes reach `gh-pages`, or manually on either
branch. It assembles the website with `play/` and `gameoflife/` from `gh-pages`,
then publishes via GitHub Pages Actions. Repository Settings → Pages must use
GitHub Actions as the build source (configured during initial publication).
Existing web applications remain sourced from `gh-pages`. Do not delete that
branch or those directories. The same workflow is installed on `gh-pages` so
reviewed website snapshots can be published before the source PR reaches
`main`. Keep the workflow copies synchronized.

The four sections are `#home`, `#install`, `#architecture`, and `#examples`.
Keep commands consistent with CMake targets and platform support. Screenshot
URLs point to pinned commits in LongMarchAssetsLFS. The home illustration is
CSS concept art, not a renderer output.

Architecture uses nested, collapsible navigation. Component heading IDs such as
`#arch-graphics` and `#arch-camera` are stable deep links: opening or refreshing
them selects the architecture tab, expands the relevant group and scrolls to
the component. The reading position updates the active directory item. On
small screens the directory becomes a bounded panel above the content.
