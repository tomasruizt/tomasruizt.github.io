clean:
	rm -rf docs

render:
	make clean
	quarto render tomas-blog && mv tomas-blog/docs .
run:
	quarto preview tomas-blog --port 3000
install:
	cd tomas-blog && quarto add r-wasm/quarto-live