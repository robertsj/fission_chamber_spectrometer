.PHONY: plots presentation clean

plots:
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
	echo "Not ready yet!"

all: plots 
