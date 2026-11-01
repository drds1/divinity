FROM python:3.12-slim

RUN useradd --create-home app
WORKDIR /home/app/divinity

COPY pyproject.toml README.md LICENSE ./
COPY divinity/ divinity/
COPY tests/ tests/
COPY benchmarks/ benchmarks/

RUN pip install --no-cache-dir ".[dev]"

USER app
CMD ["pytest"]

# docker build -f Docker/Dockerfile -t divinity .
# docker run --rm divinity
