#!/bin/bash

# EquiFlex Pro Deployment Script
set -e

echo "🚀 Deploying EquiFlex Pro..."

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
REGISTRY=${DOCKER_REGISTRY:-localhost:5000}

echo "📋 Deployment Configuration:"
echo "   Environment: $ENVIRONMENT"
echo "   Version: $VERSION"
echo "   Registry: $REGISTRY"

# Build Docker images
echo "🔨 Building Docker images..."
docker build -t $REGISTRY/equiflex-trading-engine:$VERSION --target engine .
docker build -t $REGISTRY/equiflex-api:$VERSION --target api .

# Push to registry (if not local)
if [[ $REGISTRY != "localhost:5000" ]]; then
    echo "📤 Pushing images to registry..."
    docker push $REGISTRY/equiflex-trading-engine:$VERSION
    docker push $REGISTRY/equiflex-api:$VERSION
fi

# Deploy with Docker Compose (local/staging)
if [[ $ENVIRONMENT == "local" || $ENVIRONMENT == "staging" ]]; then
    echo "🐳 Deploying with Docker Compose..."
    
    # Export environment variables
    export IMAGE_TAG=$VERSION
    export ENVIRONMENT=$ENVIRONMENT
    
    # Deploy
    docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Health check
    echo "🏥 Running health checks..."
    docker-compose ps
    
# Deploy with Kubernetes (production)
elif [[ $ENVIRONMENT == "production" ]]; then
    echo "☸️  Deploying with Kubernetes..."
    
    # Apply configurations
    kubectl apply -f infrastructure/kubernetes/namespace.yaml
    kubectl apply -f infrastructure/kubernetes/secrets.yaml
    kubectl apply -f infrastructure/kubernetes/configmap.yaml
    kubectl apply -f infrastructure/kubernetes/deployment.yaml
    kubectl apply -f infrastructure/kubernetes/service.yaml
    kubectl apply -f infrastructure/kubernetes/ingress.yaml
    
    # Wait for rollout
    echo "⏳ Waiting for deployment rollout..."
    kubectl rollout status deployment/equiflex-trading-engine -n equiflex
    kubectl rollout status deployment/equiflex-api -n equiflex
    
    # Check status
    echo "📊 Checking deployment status..."
    kubectl get pods -n equiflex
    kubectl get services -n equiflex
fi

# Run smoke tests
echo "🧪 Running smoke tests..."
python scripts/smoke_tests.py --environment $ENVIRONMENT

echo "✅ Deployment completed successfully!"

# Display access information
echo ""
echo "🌐 Access Information:"
if [[ $ENVIRONMENT == "local" ]]; then
    echo "   Web Dashboard: http://localhost:8000"
    echo "   API Documentation: http://localhost:8000/api/docs"
    echo "   Grafana: http://localhost:3000 (admin/admin)"
    echo "   Kibana: http://localhost:5601"
elif [[ $ENVIRONMENT == "production" ]]; then
    echo "   Web Dashboard: https://equiflex.yourdomain.com"
    echo "   API Documentation: https://api.equiflex.yourdomain.com/docs"
fi

echo ""
echo "📖 Monitoring:"
echo "   Check logs: docker-compose logs -f (Docker) | kubectl logs -f deployment/equiflex-trading-engine -n equiflex (K8s)"
echo "   Monitor metrics: Open Grafana dashboard"
echo "   View performance: Check /api/status endpoint"