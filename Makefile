render:
	quarto render tomas-blog && rm -rf docs && mv tomas-blog/docs .
run:
	quarto preview tomas-blog