# CV Project - Agent Notes

## Tech Stack
- **Quarto** (.qmd) renders to PDF via LaTeX
- **R** tibbles hold CV data; custom functions (`cvevents()`, `cvproj()`) emit LaTeX commands
- **Template**: `_extensions/schochastics/modern2-cv/template.tex` — checked into git with custom modifications. Do NOT reinstall from upstream (`quarto use template ...`) as it will overwrite customizations.
- **Build**: `make run` (runs `quarto preview cv.qmd`)

## Key Files
- `cv.qmd` — main CV source (YAML frontmatter + R code blocks)
- `_extensions/schochastics/modern2-cv/template.tex` — LaTeX template (customized spacing, sidebar sections, header text)
- `_extensions/schochastics/modern2-cv/styles.lua` / `para.lua` — Pandoc Lua filters

## Template Customizations Made
- Sidebar uses `\cvsidebarsection` (smaller `\large` font) instead of `\cvsection` (`\LARGE`)
- Sidebar item spacing reduced from `6pt` to `3pt`
- Sidebar icon box sizes reduced (16→12 for icons, 12→10 for bullets)
- `\cvsection` vspace reduced from `14pt` to `8pt`
- `\cvevent` post-spacing reduced from `14pt` to `8pt`
- Header says "Resume" (not "CV") — targeting US companies

## Content Guidelines
- Target audience: US-based companies (industry internships)
- Avoid redundancy between summary and detailed sections
- Use bullet points (`\begin{itemize}`) inside `\cvevent` descriptions for scannability
- Keep to one page
