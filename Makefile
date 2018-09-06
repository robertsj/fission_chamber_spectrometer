.PHONY: plots presentation proceeding clean

plots:
	mkdir -p img
	python ./code/process_cross_sections.py
	python ./code/flux_spectrum.py
	python ./code/response.py
	python ./code/shielded_responses.py
	python ./code/unfold.py

presentation:
	bash ./presentation/makeit

clean:
	rm ./presentation/roberts_animma.pdf
	rm ./presentation/roberts_animma.aux

proceeding:
	bash ./proceeding/makeit

umg:
	python ./code/unfold_umg.py
	python ./code/process_umg.py
	bash ./journal/makeit

all: plots 
