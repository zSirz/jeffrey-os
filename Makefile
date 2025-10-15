.PHONY: setup fmt lint typecheck compile smoke ci clean redlist fix-ui-imports

PYTHONPATH := src

setup: ## Installation des outils de développement
	pip install -U ruff pre-commit mypy
	pre-commit install || true

fmt: ## Formatage du code
	ruff format src/

lint: ## Linting avec exclusion vendors/icloud
	# Ruff respecte pyproject.toml (vendors exclus automatiquement)
	ruff check --fix src/

typecheck: ## Vérification des types
	mypy src/jeffrey/core --ignore-missing-imports

compile: ## Compilation uniquement du core
	# Compile uniquement le core (pas vendors)
	PYTHONPATH=src python -m compileall -q src/jeffrey

smoke: ## Tests smoke ignorant vendors/icloud
	# Smoke test ignore vendors/icloud
	PYTHONPATH=src python scripts/smoke_import.py

redlist: ## Liste courte des RED avec fichier:ligne:message
	# Liste courte des RED (syntax/indent) avec fichier:ligne:message
	PYTHONPATH=src python scripts/redlist.py

fix-ui-imports: ## Normalise les imports UI avec AST
	PYTHONPATH=src python scripts/fix_ui_imports.py

ci: lint compile smoke ## Pipeline CI complète
	@echo "✓ CI passed"

clean: ## Nettoie les fichiers temporaires
	find src -type d -name __pycache__ -exec rm -rf {} +
	find src -name "*.pyc" -delete

# Legacy targets pour compatibilité
help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

audit: ## Audit de sécurité Rust + Python
	bash scripts/audit_rust.sh || true
	pip-audit || true

start: ## Lance le système (API avec uvicorn)
	@echo "🚀 Démarrage de Jeffrey OS (API)..."
	@export SECURITY_MODE=dev && \
	uvicorn src.jeffrey.core.control.control_plane:app \
		--host 127.0.0.1 \
		--port 8000 \
		--reload &
	@sleep 2
	@echo "✅ API démarrée → http://localhost:8000/health"

stop: ## Arrête le système
	@echo "🛑 Arrêt de Jeffrey OS..."
	@pkill -f uvicorn || true
	@echo "✅ Arrêté"

test: ## Lance tous les tests
	pytest tests/ -v

# Core
up-core: ## Lance Core (NATS, Redis, PostgreSQL, Caddy)
	@echo "$$(echo -e '$(GREEN)')🚀 Starting CORE...$$(echo -e '$(NC)')"
	@cd docker/compose && docker compose -f compose.core.yml --profile core up -d
	@sleep 10
	@$(MAKE) test-core

down-core: ## Arrête Core
	@echo "$$(echo -e '$(YELLOW)')🛑 Stopping CORE...$$(echo -e '$(NC)')"
	@cd docker/compose && docker compose -f compose.core.yml --profile core down

# Observability
up-observe: ## Lance Observability
	@echo "$$(echo -e '$(GREEN)')📊 Starting OBSERVABILITY...$$(echo -e '$(NC)')"
	@cd docker/compose && docker compose -f compose.observe.yml --profile observe up -d
	@sleep 10
	@$(MAKE) test-observe

down-observe: ## Arrête Observability
	@echo "$$(echo -e '$(YELLOW)')🛑 Stopping OBSERVABILITY...$$(echo -e '$(NC)')"
	@cd docker/compose && docker compose -f compose.observe.yml --profile observe down

# All
up: up-core up-observe ## Lance tout

down: down-core down-observe ## Arrête tout

restart: down up ## Redémarre tout

# Status
status: ## Statut des services
	@echo "$$(echo -e '$(GREEN)')📊 Services Status:$$(echo -e '$(NC)')"
	@docker ps --filter "name=jeffrey-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Tests
test-core: ## Test Core
	@echo "$$(echo -e '$(GREEN)')🧪 Testing CORE...$$(echo -e '$(NC)')"
	@curl -sf http://127.0.0.1:8223/healthz && echo "  ✅ NATS OK" || echo "  ❌ NATS Failed"
	@docker exec jeffrey-redis-p2 sh -c "redis-cli -a \$$(cat /run/secrets/redis_password) ping" >/dev/null 2>&1 && \
		echo "  ✅ Redis OK" || echo "  ❌ Redis Failed"
	@docker exec jeffrey-postgres-p2 pg_isready -U jeffrey >/dev/null 2>&1 && \
		echo "  ✅ PostgreSQL OK" || echo "  ❌ PostgreSQL Failed"
	@curl -sf http://127.0.0.1:8080/healthz && echo "  ✅ Caddy OK" || echo "  ❌ Caddy Failed"

test-observe: ## Test Observability
	@echo "$$(echo -e '$(GREEN)')🧪 Testing OBSERVABILITY...$$(echo -e '$(NC)')"
	@curl -sf http://127.0.0.1:9090/-/healthy && echo "  ✅ Prometheus OK" || echo "  ❌ Prometheus Failed"
	@curl -sf http://127.0.0.1:3000/api/health && echo "  ✅ Grafana OK" || echo "  ❌ Grafana Failed"
	@curl -sf http://127.0.0.1:7777/metrics >/dev/null && echo "  ✅ NATS Exporter OK" || echo "  ❌ NATS Exporter Failed"
	@curl -sf http://127.0.0.1:9121/metrics >/dev/null && echo "  ✅ Redis Exporter OK" || echo "  ❌ Redis Exporter Failed"
	@curl -sf http://127.0.0.1:9187/metrics >/dev/null && echo "  ✅ Postgres Exporter OK" || echo "  ❌ Postgres Exporter Failed"

# Metrics
nats-metrics: ## Métriques NATS
	@curl -s http://127.0.0.1:7777/metrics | grep "^nats_" | head -10

redis-metrics: ## Métriques Redis
	@curl -s http://127.0.0.1:9121/metrics | grep "^redis_" | head -10

# URLs
urls: ## Affiche les URLs
	@echo "$$(echo -e '$(GREEN)')🌐 Service URLs:$$(echo -e '$(NC)')"
	@echo "  NATS:       http://localhost:8223"
	@echo "  Caddy:      https://localhost:8443 (self-signed)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000"
	@echo "    User: admin"
	@echo "    Pass: $$(cat docker/secrets/grafana_password 2>/dev/null || echo 'voir docker/secrets/')"
	@echo "  Redis:      redis://localhost:6380"
	@echo "  PostgreSQL: postgresql://localhost:5433/jeffrey_p2"

# Export PYTHONPATH globally
export PYTHONPATH

# NeuralBus Commands
.PHONY: nb-test nb-benchmark nb-monitor nb-stream-info nb-consumer-info

nb-test: ## Run NeuralBus tests
	@echo "$$(echo -e '$(GREEN)')🧠 Testing NeuralBus...$$(echo -e '$(NC)')"
	@pytest -v tests/test_neuralbus.py

nb-benchmark: ## Benchmark NeuralBus performance
	@echo "$$(echo -e '$(GREEN)')⚡ Benchmarking NeuralBus...$$(echo -e '$(NC)')"
	@python scripts/benchmark_neuralbus.py

nb-monitor: ## Monitor stream in real-time
	@echo "$$(echo -e '$(GREEN)')📊 Monitoring NeuralBus...$$(echo -e '$(NC)')"
	@watch -n 1 'docker run --rm --network jeffrey-network-p2 natsio/nats-box \
		nats --server nats://nats-p2:4222 stream info EVENTS'

nb-stream-info: ## Show stream information
	@docker run --rm --network jeffrey-network-p2 natsio/nats-box \
		nats --server nats://nats-p2:4222 stream info EVENTS

nb-consumer-info: ## Show consumer information
	@docker run --rm --network jeffrey-network-p2 natsio/nats-box \
		nats --server nats://nats-p2:4222 consumer info EVENTS WORKERS

nb-pub-test: ## Publish test event
	@docker run --rm --network jeffrey-network-p2 natsio/nats-box \
		nats --server nats://nats-p2:4222 pub events.test \
		'{"meta":{"type":"test.event","tenant_id":"test","priority":"normal","jeffrey_copyright":"proprietary-jeffrey-os"},"data":{"msg":"test"}}'

# Production Load Testing Commands
.PHONY: load-quick load-stress load-chaos load-soak load-all

# Load test namespace variable
LOAD_NS := $(shell date +test_%H%M%S)

load-setup: ## Setup for load testing
	@echo "$$(echo -e '$(YELLOW)')🔧 Setting up load testing environment...$(NC)"
	@$(PIP) install -e . --quiet
	@mkdir -p logs .nats data/metrics
	@echo "$$(echo -e '$(GREEN)')✅ Load test setup complete!$(NC)"

load-quick: load-setup ## Run quick validation (3 min)
	@echo "$$(echo -e '$(YELLOW)')🚀 Running quick test...$(NC)"
	@NB_NS=$(LOAD_NS) $(PYTHON) scripts/generate_load.py --phase quick

load-stress: load-setup ## Run stress test (6 min)
	@echo "$$(echo -e '$(YELLOW)')🚀 Running stress test...$(NC)"
	@NB_NS=$(LOAD_NS) $(PYTHON) scripts/generate_load.py --phase stress --ml

load-chaos: load-setup ## Run chaos test (6 min)
	@echo "$$(echo -e '$(YELLOW)')🔥 Running chaos test...$(NC)"
	@NB_NS=$(LOAD_NS) $(PYTHON) scripts/generate_load.py --phase chaos --corruption

load-all: load-setup ## Run all progressive tests
	@echo "$$(echo -e '$(YELLOW)')🧪 Running all tests...$(NC)"
	@$(PYTHON) scripts/launch_soak_test.py --non-interactive

load-soak: load-setup ## Run 2-hour production soak test
	@echo "$$(echo -e '$(YELLOW)')⏰ Starting 2-hour soak test...$(NC)"
	@echo "$$(echo -e '$(YELLOW)')📊 Metrics: http://localhost:8000/metrics$(NC)"
	@NB_NS=$(LOAD_NS) $(PYTHON) scripts/generate_load.py --phase soak --ml --monitor

load-soak-ci: load-setup ## Run soak test in CI mode
	@NB_NS=$(LOAD_NS) $(PYTHON) scripts/generate_load.py --phase soak --ml --non-interactive

nats-local-start: ## Start local NATS for load testing
	@$(PYTHON) scripts/nats_manager.py start --namespace $(LOAD_NS)

nats-local-stop: ## Stop local NATS
	@$(PYTHON) scripts/nats_manager.py stop

nats-local-status: ## Check local NATS status
	@$(PYTHON) scripts/nats_manager.py status
