.PHONY : build develop test

all: develop

build: clean_pyc
	python setup.py build_ext --inplace

develop: build
	python -m pip install -e . -v  --no-build-isolation --no-use-pep517

test:
	py.test --pyargs fast_upfirdn --cov=fast_upfirdn --cov-report term-missing --cov-report html
