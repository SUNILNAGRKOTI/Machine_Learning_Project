// Chart instances
let comparisonChart = null;
let featureChart = null;
let trendChart = null;

// Store prediction history
let predictionHistory = [];

// API endpoint
const API_URL = 'http://127.0.0.1:5000';

// Register Chart.js plugin for datalabels
Chart.register(ChartDataLabels);

// --- Event Listener ---
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get input values
    const sleep = parseFloat(document.getElementById('sleep').value);
    const study = parseFloat(document.getElementById('study').value);
    const screen = parseFloat(document.getElementById('screen').value);
    const activity = parseFloat(document.getElementById('activity').value);
    const caffeine = parseFloat(document.getElementById('caffeine').value);
    
    // Simple validation (more advanced validation can be added)
    if (isNaN(sleep) || isNaN(study) || isNaN(screen) || isNaN(activity) || isNaN(caffeine)) {
        alert('Please ensure all fields have valid numbers!');
        return;
    }
    
    try {
        // Show processing state
        document.getElementById('emptyState').classList.add('hidden');
        document.getElementById('resultsCard').classList.remove('hidden');
        document.getElementById('visualizationsSection').classList.add('hidden');
        const statusBadge = document.getElementById('statusBadge');
        statusBadge.className = 'status-badge';
        statusBadge.innerHTML = '<span class="status-icon">⏳</span><span class="status-text">Processing...</span>';
        
        // Prepare data for API
        const inputData = { sleep, study, screen, activity: activity / 60, caffeine }; // Convert activity to hours for ML model if it expects hours
        
        // Call backend API
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(inputData)
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Store in history
            predictionHistory.push({
                date: new Date(),
                inputs: inputData,
                score: data.prediction.score,
                level: data.prediction.level
            });
            
            // Display results
            displayResults(data, sleep, study, screen, activity, caffeine);
            
            // Generate visualizations
            generateCharts(data);

            // Show visualizations section
            document.getElementById('visualizationsSection').classList.remove('hidden');

            // Automatically switch to the Feature Importance tab on first load/prediction
            showChart(null, 'featureChart');
        } else {
            alert('Prediction Error: ' + (data.error || 'Unknown error'));
            resetView();
        }
    } catch (error) {
        alert('Connection Error: Make sure the Flask API is running on ' + API_URL + '\n\nError: ' + error.message);
        resetView();
        console.error('Error:', error);
    }
});

// --- Utility Functions ---

function resetView() {
    document.getElementById('resultsCard').classList.add('hidden');
    document.getElementById('emptyState').classList.remove('hidden');
    document.getElementById('visualizationsSection').classList.add('hidden');
}

// Display results
function displayResults(data, sleep, study, screen, activity, caffeine) {
    // Update status badge
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.className = 'status-badge ' + data.prediction.level.toLowerCase();
    
    statusBadge.innerHTML = `
        <span class="status-icon">${data.prediction.emoji}</span>
        <span class="status-text">${data.prediction.level} PRODUCTIVITY</span>
    `;
    
    // Update score
    document.getElementById('scoreValue').textContent = data.prediction.score;
    
    // Update summary - using original input format for display
    document.getElementById('summarySleep').textContent = sleep + 'h';
    document.getElementById('summaryStudy').textContent = study + 'h';
    document.getElementById('summaryScreen').textContent = screen + 'h';
    document.getElementById('summaryActivity').textContent = activity + ' min';
    document.getElementById('summaryCaffeine').textContent = caffeine + ' cup(s)';
    document.getElementById('summaryConfidence').textContent = data.confidence + '%';
    
    // Update recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = data.recommendations
        .sort((a, b) => (a.priority === 'high' ? -1 : 1)) // Sort high priority first
        .map(rec => '<li>' + rec.message + '</li>')
        .join('');
}

// Show chart function
function showChart(event, chartId) {
    // Remove active class from all charts
    document.querySelectorAll('.chart-wrapper').forEach(wrapper => {
        wrapper.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to selected chart
    document.getElementById(chartId).classList.add('active');
    
    // Add active class to clicked tab
    if (event) {
        event.target.classList.add('active');
    } else {
        // If no event (i.e., called from prediction), find the correct tab
        document.querySelector(`.tab-btn[onclick*="${chartId}"]`).classList.add('active');
    }
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    resetView();
}

function downloadReport() {
    // Implementation is good, kept as is.
    if (predictionHistory.length === 0) {
        alert('No predictions to download yet!');
        return;
    }
    
    const lastPrediction = predictionHistory[predictionHistory.length - 1];
    const inputs = lastPrediction.inputs;
    const score = lastPrediction.score;
    const level = lastPrediction.level;
    const confidence = document.getElementById('summaryConfidence').textContent;
    const recommendations = document.getElementById('recommendationsList').innerText;

    
    let reportContent = 'PRODUCTIVITY PREDICTION REPORT\n';
    reportContent += '==============================\n\n';
    reportContent += 'Date: ' + new Date().toLocaleString() + '\n\n';
    reportContent += 'INPUT DATA:\n';
    reportContent += `- Sleep Duration: ${document.getElementById('summarySleep').textContent}\n`;
    reportContent += `- Study Hours: ${document.getElementById('summaryStudy').textContent}\n`;
    reportContent += `- Screen Time: ${document.getElementById('summaryScreen').textContent}\n`;
    reportContent += `- Physical Activity: ${document.getElementById('summaryActivity').textContent}\n`;
    reportContent += `- Caffeine Intake: ${document.getElementById('summaryCaffeine').textContent}\n\n`;
    reportContent += 'PREDICTION:\n';
    reportContent += `- Productivity Score: ${score} / 100\n`;
    reportContent += `- Level: ${level}\n`;
    reportContent += `- Model Confidence: ${confidence}\n\n`;
    reportContent += 'RECOMMENDATIONS:\n' + recommendations + '\n\n';
    reportContent += 'Generated by Productivity Prediction System\n';
    reportContent += 'Machine Learning Models: Random Forest Regressor\n';
    
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(reportContent));
    element.setAttribute('download', 'productivity_report_' + Date.now() + '.txt');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

// --- Chart Generation ---

function generateCharts(data) {
    // Feature Importance Chart - Recommended to show this first
    generateFeatureChart(data.feature_importance);
    
    // Comparison Chart
    generateComparisonChart(data.prediction.score);
    
    // Trend Chart
    generateTrendChart();
}

// Comparison Chart: Your Score vs Benchmarks
function generateComparisonChart(score) {
    const ctx = document.getElementById('comparisonCanvas').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Your Score', 'Excellent (>80)', 'Good (50-80)', 'Average (20-50)'],
            datasets: [{
                label: 'Productivity Score',
                data: [score, 90, 65, 35],
                backgroundColor: [
                    'rgba(37, 99, 235, 0.9)', // Your Score (Primary)
                    'rgba(16, 185, 129, 0.8)', // Excellent (Success)
                    'rgba(245, 158, 11, 0.8)', // Good (Warning)
                    'rgba(239, 68, 68, 0.8)' // Average (Danger)
                ],
                borderColor: 'rgba(255, 255, 255, 1)',
                borderWidth: 1,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Your Score vs. Benchmark Levels' },
                datalabels: {
                    anchor: 'end',
                    align: 'top',
                    formatter: Math.round,
                    color: '#0f172a',
                    font: { weight: 'bold' }
                }
            },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: 'Productivity Score' } },
                x: { grid: { display: false } }
            }
        }
    });
}

// Feature Importance Chart: Doughnut chart is great for this
function generateFeatureChart(importance) {
    const ctx = document.getElementById('featureCanvas').getContext('2d');
    
    if (featureChart) {
        featureChart.destroy();
    }
    
    const labels = ['Sleep Duration', 'Study Hours', 'Screen Time', 'Physical Activity', 'Caffeine Intake'];
    const data = [
        importance.sleep || 30, // Default values if importance is missing
        importance.study || 22,
        importance.screen || 19,
        importance.activity || 21,
        importance.caffeine || 8
    ];
    
    featureChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#2563eb', // Blue
                    '#8b5cf6', // Violet
                    '#f59e0b', // Amber
                    '#10b981', // Green
                    '#ef4444' // Red
                ],
                hoverOffset: 10,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'right' },
                title: { display: true, text: 'Impact of Factors on Prediction' },
                datalabels: {
                    formatter: (value, context) => {
                        return context.chart.data.labels[context.dataIndex] + ': ' + value + '%';
                    },
                    color: '#fff',
                    font: { weight: 'bold' },
                    textAlign: 'center'
                }
            }
        }
    });
}

// Trend Chart: Last 7 predictions
function generateTrendChart() {
    const ctx = document.getElementById('trendCanvas').getContext('2d');
    
    if (trendChart) {
        trendChart.destroy();
    }
    
    const historyData = predictionHistory.slice(-7);
    const labels = historyData.map((item, index) => {
        const date = new Date(item.date);
        return `${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;
    });
    const data = historyData.map(item => item.score);
    
    if (data.length === 0) {
        labels.push('Today');
        data.push(0);
    }
    
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Productivity Trend (Score / 100)',
                data: data,
                borderColor: 'rgba(37, 99, 235, 1)',
                backgroundColor: 'rgba(37, 99, 235, 0.15)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointBackgroundColor: 'rgba(37, 99, 235, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Last 7 Prediction Scores' }
            },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: 'Score' } },
                x: { grid: { display: false } }
            }
        }
    });
}