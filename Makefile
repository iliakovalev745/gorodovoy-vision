SHELL := /bin/bash
.PHONY: install run stop purge
install:
	@python -m venv .venv
	@. .venv/bin/activate && python -m pip install --upgrade pip wheel && python -m pip install -r requirements.txt
	@test -f .env || cp .env.example .env
run:
	@mkdir -p logs
	@nohup ./.venv/bin/python -m app > logs/app.log 2>&1 & echo $$! > .run.pid
	@sleep 2
	@echo "Started. Logs: tail -f logs/app.log"
stop:
	@if [ -f .run.pid ]; then kill $$(cat .run.pid) 2>/dev/null || true; rm -f .run.pid; fi
purge: stop
	@rm -rf .venv __pycache__ app/__pycache__ vision/__pycache__ logs .run.pid
	@echo "Removed generated files; .env kept."
