
packages = python_speech_features scipy numpy


run: dependencies
	python3 genre-predictor.py

clean:
	rm -rf my.dat
	pip uninstall -y $(packages)

dependencies:
	pip install $(packages)
