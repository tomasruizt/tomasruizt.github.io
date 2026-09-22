To publish, run the command `make render-for-publish`. This will overwrite the `docs/` directory with the rendered content. The `docs/` directory is what is served by GitHub Pages.

Standalone reports live in `reports/` and are copied into `docs/reports/` during publishing.
The [B200 DFlash benchmark report](https://tomasruizt.github.io/reports/b200-dflash/) includes both draft lengths, all three models, and linked logs.

# Installation

1. Install Quarto: https://quarto.org/docs/get-started/
2. Install quarto dependencies for compilation:
```shell
make install
```
3. (Optional) Install Quarto VSCode Extension.
