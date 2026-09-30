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

The home, installation and samples sections use `#home`, `#install` and
`#examples`. Architecture lives in independent static documents under
`reference/`. Legacy `#architecture` and `#arch-*` links redirect to the
corresponding documentation page. See `docs/reference/README.md` for the
source review process and document build command.
