FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gmsh && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir gmsh meshio numpy matplotlib
WORKDIR /work
COPY . /work
