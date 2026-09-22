# B200 DFlash benchmark report

- Extract the entire ZIP and open [index.html](index.html). Keep the directory structure intact so Server and Bench links work.
- Includes both K settings, all three models, server/benchmark logs, per-point summaries and configurations, plots, environment versions, and validation.
- Model selection stays constant when switching K; no JavaScript or internet connection is required to view the report.
- Excludes raw profiler traces and per-request datasets; this is a report bundle, not the full experiment archive.

## Share on GitHub

- **Quickest:** attach the ZIP to a GitHub release and share the download link. Readers extract it and open index.html.
- **Best browsing experience:** publish this folder with GitHub Pages, then share the resulting website URL. GitHub repository file previews do not display the interactive HTML report.
- For a dedicated report repository: copy this folder's contents into docs/, commit and push, then select Settings → Pages → Deploy from a branch → main → /docs. Keep .nojekyll.
- For an existing Pages site, put this folder in a subdirectory of its publishing source.

[GitHub Pages setup](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
