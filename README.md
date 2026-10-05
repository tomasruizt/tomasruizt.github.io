To publish, run the command `make render-for-publish`. This will overwrite the `docs/` directory with the rendered content. The `docs/` directory is what is served by GitHub Pages.

Standalone reports live in `reports/` and are copied into `docs/reports/` during publishing.
The [B200 DFlash benchmark report](https://tomasruizt.github.io/reports/b200-dflash/) includes both draft lengths, all three models, and linked logs.
The [Nemotron DSpark AV benchmark](https://tomasruizt.github.io/reports/nemotron-3.5-dspark-av/) compares K=7 and K=15 with AV off/on on one B300.

# Installation

1. Install Quarto: https://quarto.org/docs/get-started/
2. Install quarto dependencies for compilation:
```shell
make install
```
3. (Optional) Install Quarto VSCode Extension.
