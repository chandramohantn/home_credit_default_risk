# Portable Workspace Makefile
# Copy this to your project folder alongside docker-compose.yml

.PHONY: help up down logs shell

# Set your preferred workspace image here
# Options: ml-workspace, dl-workspace-cpu, dl-workspace-gpu, llm-workspace
export WORKSPACE_IMAGE ?= ml-workspace:latest

help:
	@echo "Portable AI/ML Workspace"
	@echo "========================"
	@echo "Using Image: $(WORKSPACE_IMAGE)"
	@echo ""
	@echo "Commands:"
	@echo "  make up      - Start workspace in background (Dozzle at :9999)"
	@echo "  make down    - Stop workspace"
	@echo "  make logs    - Follow workspace logs"
	@echo "  make shell   - Open a bash shell in the running container"

up:
	docker compose up -d workspace dozzle
	@echo "Workspace started! JupyterLab at http://localhost:8888"
	@echo "Log Viewer at http://localhost:9999"

down:
	docker compose down

logs:
	docker compose logs -f workspace

shell:
	docker compose exec workspace bash
