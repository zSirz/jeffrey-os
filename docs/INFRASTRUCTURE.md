# Jeffrey OS Infrastructure

## Architecture

Jeffrey OS est une architecture de traitement émotionnel et cognitif distribuée basée sur Docker avec monitoring intégré.

### Components
- **API**: FastAPI on port 8000
- **Database**: Redis on port 6379 (distributed locks & caching)
- **Monitoring**: Prometheus (9090) + Grafana (3000)
- **Processing**: DreamEngine with auto-learning

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Grafana       │    │   Prometheus    │
│   (Future)      │    │   :3000         │    │   :9090         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       └───────┬───────────────┘
         │                               │
┌─────────────────────────────────────────┴─────────────────────────┐
│                     Jeffrey API :8000                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │   FastAPI       │  │  DreamEngine    │  │   NeuralBus     │   │
│  │   Endpoints     │  │   Progressive   │  │   Events        │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                    ┌─────────────────┐
                    │     Redis       │
                    │     :6379       │
                    │  (Locks & Cache)│
                    └─────────────────┘
```

## Components Status

✅ **Infrastructure**
- [x] Docker Compose setup
- [x] Environment configuration (.env)
- [x] Health checks (healthz/readyz)
- [x] Service dependencies

✅ **Monitoring & Observability**
- [x] Prometheus metrics collection
- [x] Grafana dashboards
- [x] Custom Jeffrey metrics
- [x] Service health monitoring
- [x] Performance tracking

✅ **Core Processing**
- [x] DreamEngine progressive
- [x] Auto-learning batch optimization
- [x] Redis distributed locks
- [x] Idempotence with persistent tracking
- [x] Dead Letter Queue (DLQ)
- [x] Circuit breakers & timeouts

✅ **API Layer**
- [x] FastAPI framework
- [x] Pydantic validation
- [x] OpenAPI documentation
- [x] CORS middleware
- [x] Metrics endpoints

## Accessing Services

### Main Interfaces
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/jeffrey2024secure)
- **Prometheus Metrics**: http://localhost:9090
- **Raw Metrics**: http://localhost:8000/metrics

### Key API Endpoints
- `GET /healthz` - Liveness probe
- `GET /readyz` - Readiness probe with dependency checks
- `GET /metrics` - Prometheus metrics
- `POST /api/v1/dream/toggle` - Enable/disable DreamEngine
- `POST /api/v1/dream/run` - Force dream consolidation
- `GET /api/v1/dream/status` - DreamEngine statistics

## Configuration

### Environment Variables
```env
# Redis Configuration
REDIS_PASSWORD=jeffrey_redis_secure_2024
REDIS_URL=redis://:jeffrey_redis_secure_2024@redis:6379/0

# Grafana Configuration
GF_SECURITY_ADMIN_PASSWORD=jeffrey2024secure

# Security
JEFFREY_SECRET_KEY=your_secret_key_here
SECURITY_MODE=dev
LOG_LEVEL=INFO
```

### Docker Compose Services
- **jeffrey-api**: Main application (CPU: 1.0, Memory: 1G)
- **redis**: Distributed locks and caching
- **prometheus**: Metrics collection and storage
- **grafana**: Visualization and alerting

## Monitoring & Metrics

### Custom Metrics
- `jeffrey_dream_quality` - Dream consolidation quality score
- `jeffrey_dream_batch_size` - Current batch processing size
- `jeffrey_dream_dlq_size` - Dead letter queue failures
- `jeffrey_dream_runs_total` - Total dream consolidation runs
- `jeffrey_emotion_ml_confidence_sum` - Sum of ML confidence scores
- `jeffrey_emotion_ml_confidence_count` - Count of emotion predictions

### Grafana Dashboards
- **Jeffrey Brain Complete**: Real-time brain activity monitoring
- **System Overview**: Infrastructure health and performance
- **Dream Analytics**: DreamEngine performance and quality

### Alerts
- No dream runs in 24h
- High DLQ size (>10 entries)
- High latency (P95 > 500ms)
- Queue saturation (>85%)
- Service down alerts

## Development Workflow

### Quick Start
```bash
# Clone and setup
git clone <repository>
cd Jeffrey_OS

# Start infrastructure
docker-compose up -d

# Verify health
./scripts/test_infrastructure.sh

# View logs
docker-compose logs -f jeffrey-api
```

### Testing
```bash
# Infrastructure test
./scripts/test_infrastructure.sh

# API test
curl http://localhost:8000/healthz

# Dream engine test
curl -X POST http://localhost:8000/api/v1/dream/run?force=true

# Metrics check
curl http://localhost:8000/metrics | grep jeffrey_
```

### Development Mode
```bash
# For development with hot reload
PYTHONPATH=src uvicorn jeffrey.interfaces.bridge.api:app --reload --host 0.0.0.0 --port 8000
```

## Security

### Authentication
- Grafana: Admin credentials in .env
- Redis: Password-protected
- API: Secret key based (development mode)

### Network Security
- Services isolated in Docker network
- Internal communication only
- External access via defined ports only

### Data Security
- Environment variables for secrets
- Redis password protection
- No plaintext credentials in code

## Performance

### Current Benchmarks
- API response time: <100ms (P95)
- Dream consolidation: <60s timeout
- Memory pagination: 50 items/page
- Auto-learning batch optimization

### Scaling Considerations
- Stateless API design
- Redis for distributed coordination
- Configurable resource limits
- Horizontal scaling ready

## Next Priorities

### Phase 1: Core Functionality
1. **Memory Storage**: Implement persistent memory store (PostgreSQL/MongoDB)
2. **Missing Endpoints**: Create `/emotion/detect`, `/thought/process`
3. **Real Data**: Connect to actual emotion detection models
4. **WebSocket**: Real-time updates for frontend

### Phase 2: User Interface
1. **Frontend**: React/Vue dashboard
2. **Authentication**: JWT-based user system
3. **Multi-tenancy**: User isolation and data separation
4. **API Gateway**: Rate limiting and routing

### Phase 3: Advanced Features
1. **Machine Learning**: Advanced emotion and thought models
2. **Analytics**: Historical analysis and insights
3. **Integrations**: External APIs and webhooks
4. **Mobile**: Mobile app for emotion tracking

### Phase 4: Production Hardening
1. **High Availability**: Multi-node deployment
2. **Backup/Recovery**: Data persistence and recovery
3. **Security**: Advanced auth, audit logging
4. **Performance**: Caching, optimization

## Troubleshooting

### Common Issues

**Service won't start**
```bash
# Check logs
docker-compose logs <service-name>

# Verify environment
docker-compose config

# Restart services
docker-compose restart
```

**Metrics not appearing**
```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Verify Prometheus config
docker-compose logs prometheus
```

**Redis connection issues**
```bash
# Test Redis connectivity
docker exec jeffrey_os-redis-1 redis-cli -a jeffrey_redis_secure_2024 ping

# Check password configuration
grep REDIS_PASSWORD .env
```

**Grafana login issues**
```bash
# Check admin password
grep GF_SECURITY_ADMIN_PASSWORD .env

# Reset Grafana data
docker-compose down
docker volume rm jeffrey_os_grafana_data
docker-compose up -d
```

## Support

### Logs Location
- Application: `docker-compose logs jeffrey-api`
- Monitoring: `docker-compose logs prometheus grafana`
- Infrastructure: `docker-compose logs redis`

### Health Checks
- API Health: `curl http://localhost:8000/healthz`
- Readiness: `curl http://localhost:8000/readyz`
- Prometheus: `curl http://localhost:9090/-/healthy`
- Grafana: `curl http://localhost:3000/api/health`

### Performance Monitoring
- Grafana dashboards for real-time metrics
- Prometheus for historical data
- Dream engine statistics via API
- Auto-debug reports via `/api/v1/brain/auto-debug`