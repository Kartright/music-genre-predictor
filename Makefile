
packages = python_speech_features scipy numpy


run: dependencies
	python3 genre-predictor.py

clean:
	rm -rf my.dat my.custom-dat

purge: clean
	pip uninstall -y $(packages)

dependencies:
	pip install $(packages)
