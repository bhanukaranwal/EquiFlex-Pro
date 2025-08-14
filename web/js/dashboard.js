// Dashboard JavaScript
let performanceChart;
let greeksChart;
let websocket;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadDashboardData();
    connectWebSocket();
    
    // Set up periodic refresh
    setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
    
    // Set up order type change handler
    document.getElementById('order-type').addEventListener('change', function() {
        const priceField = document.getElementById('price-field');
        if (this.value === 'limit') {
            priceField.style.display = 'block';
        } else {
            priceField.style.display = 'none';
        }
    });
});

// Initialize charts
function initializeCharts() {
    // Performance Chart
    const perfCtx = document.getElementById('performance-chart').getContext('2d');
    performanceChart = new Chart(perfCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#4e73df',
                backgroundColor: 'rgba(78, 115, 223, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Greeks Chart
    const greeksCtx = document.getElementById('greeks-chart').getContext('2d');
    greeksChart = new Chart(greeksCtx, {
        type: 'radar',
        data: {
            labels: ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
            datasets: [{
                label: 'Greeks Exposure',
                data: [0, 0, 0, 0, 0],
                borderColor: '#1cc88a',
                backgroundColor: 'rgba(28, 200, 138, 0.2)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Load dashboard data
async function loadDashboardData() {
    try {
        // Load portfolio data
        await loadPortfolioData();
        
        // Load positions
        await loadPositions();
        
        // Load strategies
        await loadStrategies();
        
        // Load orders
        await loadOrders();
        
        // Load risk metrics
        await loadRiskMetrics();
        
        // Load Greeks
        await loadGreeks();
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data');
    }
}

// Load portfolio data
async function loadPortfolioData() {
    try {
        const response = await axios.get('/api/portfolio', {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        const data = response.data;
        const summary = data.summary;
        
        // Update metrics
        document.getElementById('portfolio-value').textContent = 
            '$' + summary.total_value.toLocaleString(undefined, {minimumFractionDigits: 2});
        
        document.getElementById('total-return').textContent = 
            (summary.total_return * 100).toFixed(2) + '%';
        
        document.getElementById('max-drawdown').textContent = 
            (summary.max_drawdown * 100).toFixed(2) + '%';
        
        document.getElementById('sharpe-ratio').textContent = 
            summary.sharpe_ratio.toFixed(2);
        
        // Update performance chart (simplified)
        updatePerformanceChart(summary.total_value);
        
    } catch (error) {
        console.error('Error loading portfolio data:', error);
    }
}

// Load positions
async function loadPositions() {
    try {
        const response = await axios.get('/api/portfolio', {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        const positions = response.data.positions;
        const tbody = document.getElementById('positions-table');
        tbody.innerHTML = '';
        
        positions.forEach(position => {
            const row = document.createElement('tr');
            const pnlClass = position['Unrealized P&L'] >= 0 ? 'positive' : 'negative';
            
            row.innerHTML = `
                <td><strong>${position.Symbol}</strong></td>
                <td>${position.Quantity}</td>
                <td>$${position['Market Value'].toLocaleString()}</td>
                <td class="${pnlClass}">$${position['Unrealized P&L'].toFixed(2)}</td>
            `;
            tbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error loading positions:', error);
    }
}

// Load strategies
async function loadStrategies() {
    try {
        const response = await axios.get('/api/strategies', {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        const strategies = response.data.strategies;
        const container = document.getElementById('strategies-list');
        container.innerHTML = '';
        
        Object.values(strategies).forEach(strategy => {
            const strategyDiv = document.createElement('div');
            strategyDiv.className = 'strategy-item';
            
            const statusBadge = strategy.is_running ? 
                '<span class="badge bg-success">Running</span>' : 
                '<span class="badge bg-secondary">Stopped</span>';
            
            strategyDiv.innerHTML = `
                <div>
                    <strong>${strategy.name}</strong>
                    <br>
                    <small>Allocation: ${(strategy.allocation * 100).toFixed(1)}%</small>
                </div>
                <div class="strategy-status">
                    ${statusBadge}
                    <button class="btn btn-sm btn-outline-primary ms-2" 
                            onclick="toggleStrategy('${strategy.name}')">
                        Toggle
                    </button>
                </div>
            `;
            
            container.appendChild(strategyDiv);
        });
        
    } catch (error) {
        console.error('Error loading strategies:', error);
    }
}

// Load orders
async function loadOrders() {
    try {
        const response = await axios.get('/api/orders', {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        const orders = [...response.data.active_orders, ...response.data.order_history.slice(-10)];
        const tbody = document.getElementById('orders-table');
        tbody.innerHTML = '';
        
        orders.forEach(order => {
            const row = document.createElement('tr');
            const statusClass = order.status === 'filled' ? 'success' : 
                               order.status === 'cancelled' ? 'danger' : 'warning';
            
            row.innerHTML = `
                <td>${order.symbol}</td>
                <td><span class="badge bg-${order.side === 'buy' ? 'success' : 'danger'}">${order.side.toUpperCase()}</span></td>
                <td>${order.quantity}</td>
                <td><span class="badge bg-${statusClass}">${order.status.toUpperCase()}</span></td>
                <td>${new Date(order.timestamp).toLocaleTimeString()}</td>
            `;
            tbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error loading orders:', error);
    }
}

// Load risk metrics
async function loadRiskMetrics() {
    try {
        const response = await axios.get('/api/risk', {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        const riskData = response.data.risk_data.risk_metrics;
        const container = document.getElementById('risk-metrics');
        container.innerHTML = '';
        
        const metrics = [
            { label: 'VaR (95%)', value: `$${riskData.var_95.toLocaleString()}` },
            { label: 'Max Drawdown', value: `${(riskData.max_drawdown * 100).toFixed(2)}%` },
            { label: 'Sharpe Ratio', value: riskData.sharpe_ratio.toFixed(2) },
            { label: 'Leverage', value: `${riskData.leverage_ratio.toFixed(2)}x` }
        ];
        
        metrics.forEach(metric => {
            const metricDiv = document.createElement('div');
            metricDiv.className = 'risk-metric';
            metricDiv.innerHTML = `
                <span>${metric.label}</span>
                <span class="risk-value">${metric.value}</span>
            `;
            container.appendChild(metricDiv);
        });
        
    } catch (error) {
        console.error('Error loading risk metrics:', error);
    }
}

// Load Greeks
async function loadGreeks() {
    try {
        const response = await axios.get('/api/portfolio/greeks', {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        const greeks = response.data.greeks;
        
        // Update Greeks chart
        greeksChart.data.datasets[0].data = [
            Math.abs(greeks.delta),
            Math.abs(greeks.gamma),
            Math.abs(greeks.theta),
            Math.abs(greeks.vega),
            Math.abs(greeks.rho)
        ];
        greeksChart.update();
        
    } catch (error) {
        console.error('Error loading Greeks:', error);
    }
}

// Update performance chart
function updatePerformanceChart(value) {
    const now = new Date();
    const timeLabel = now.toLocaleTimeString();
    
    performanceChart.data.labels.push(timeLabel);
    performanceChart.data.datasets[0].data.push(value);
    
    // Keep only last 50 points
    if (performanceChart.data.labels.length > 50) {
        performanceChart.data.labels.shift();
        performanceChart.data.datasets[0].data.shift();
    }
    
    performanceChart.update('none');
}

// WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function() {
        console.log('WebSocket connected');
        document.getElementById('system-status').textContent = 'Online';
        document.getElementById('system-status').className = 'badge bg-success';
    };
    
    websocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.type === 'portfolio_update') {
            updatePortfolioMetrics(data.data);
        }
    };
    
    websocket.onclose = function() {
        console.log('WebSocket disconnected');
        document.getElementById('system-status').textContent = 'Offline';
        document.getElementById('system-status').className = 'badge bg-danger';
        
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

// Update portfolio metrics from WebSocket
function updatePortfolioMetrics(data) {
    document.getElementById('portfolio-value').textContent = 
        '$' + data.total_value.toLocaleString(undefined, {minimumFractionDigits: 2});
    
    document.getElementById('total-return').textContent = 
        (data.total_return * 100).toFixed(2) + '%';
    
    // Flash update animation
    document.getElementById('portfolio-value').parentElement.classList.add('update-flash');
    setTimeout(() => {
        document.getElementById('portfolio-value').parentElement.classList.remove('update-flash');
    }, 500);
    
    // Update chart
    updatePerformanceChart(data.total_value);
}

// Toggle strategy
async function toggleStrategy(strategyName) {
    try {
        const response = await axios.post(`/api/strategies/${strategyName}/toggle`, {}, {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        showSuccess(`Strategy ${strategyName} ${response.data.action}`);
        await loadStrategies(); // Refresh strategies list
        
    } catch (error) {
        console.error('Error toggling strategy:', error);
        showError('Failed to toggle strategy');
    }
}

// Submit order
async function submitOrder() {
    try {
        const orderData = {
            symbol: document.getElementById('order-symbol').value,
            side: document.getElementById('order-side').value,
            quantity: parseFloat(document.getElementById('order-quantity').value),
            order_type: document.getElementById('order-type').value
        };
        
        if (orderData.order_type === 'limit') {
            orderData.price = parseFloat(document.getElementById('order-price').value);
        }
        
        const response = await axios.post('/api/orders', orderData, {
            headers: { Authorization: `Bearer ${getAuthToken()}` }
        });
        
        showSuccess('Order submitted successfully');
        
        // Close modal and refresh orders
        const modal = bootstrap.Modal.getInstance(document.getElementById('orderModal'));
        modal.hide();
        await loadOrders();
        
    } catch (error) {
        console.error('Error submitting order:', error);
        showError('Failed to submit order');
    }
}

// Refresh positions
async function refreshPositions() {
    await loadPositions();
    showSuccess('Positions refreshed');
}

// Utility functions
function getAuthToken() {
    // In a real application, this would be stored securely
    return localStorage.getItem('auth_token') || 'demo_token';
}

function showSuccess(message) {
    // Simple toast notification (you could use a proper toast library)
    const toast = document.createElement('div');
    toast.className = 'alert alert-success position-fixed';
    toast.style.top = '20px';
    toast.style.right = '20px';
    toast.style.zIndex = '9999';
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        document.body.removeChild(toast);
    }, 3000);
}

function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'alert alert-danger position-fixed';
    toast.style.top = '20px';
    toast.style.right = '20px';
    toast.style.zIndex = '9999';
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        document.body.removeChild(toast);
    }, 5000);
}