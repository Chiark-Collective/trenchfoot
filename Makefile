# ABOUTME: Makefile for trenchfoot development tasks.
# ABOUTME: Provides targets for testing, mesh generation, and other common operations.

.PHONY: test dump-meshes dump-meshes-volumetric clean-meshes

# Default data directory for generated meshes
DATA_DIR := data/scenarios

# Run tests
test:
	uv run pytest -rs

# Generate scenario meshes (surface only, with previews) for inspection
dump-meshes:
	@mkdir -p $(DATA_DIR)
	uv run python -m trenchfoot.generate_scenarios \
		--out $(DATA_DIR) \
		--preview \
		--skip-volumetric
	@echo "Meshes written to $(DATA_DIR)"

# Generate scenario meshes including volumetric (requires gmsh)
dump-meshes-volumetric:
	@mkdir -p $(DATA_DIR)
	uv run python -m trenchfoot.generate_scenarios \
		--out $(DATA_DIR) \
		--preview \
		--volumetric \
		--lc 0.4
	@echo "Meshes (including volumetric) written to $(DATA_DIR)"

# Remove generated mesh data
clean-meshes:
	rm -rf $(DATA_DIR)
	@echo "Cleaned $(DATA_DIR)"
