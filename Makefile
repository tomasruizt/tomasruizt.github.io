clean:
	rm -rf docs

render-for-publish:
	make clean
	quarto render tomas-blog && mv tomas-blog/docs .
	python normalize_html.py
	cp -a reports docs/reports
preview:
	quarto preview tomas-blog --port 3000
install:
	cd tomas-blog && quarto add r-wasm/quarto-live
