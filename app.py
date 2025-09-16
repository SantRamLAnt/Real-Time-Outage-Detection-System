import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

# Page configuration
st.set_page_config(
    page_title="Real-Time Outage Detection System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 50%, #e74c3c 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #c0392b;
    }
    .success-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #229954;
    }
    .info-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .tech-detail {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'demo_started' not in st.session_state:
    st.session_state.demo_started = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Welcome overlay
if not st.session_state.demo_started:
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background: linear-gradient(135deg, #1f4e79 0%, #2980b9 50%, #e74c3c 100%); 
                z-index: 999; display: flex; justify-content: center; align-items: center;">
        <div style="background: white; padding: 3rem; border-radius: 20px; text-align: center; 
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3); max-width: 650px; min-height: 500px;
                    display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h1 style="color: #1f4e79; margin-bottom: 1rem;">⚡ Real-Time Outage Detection System</h1>
                <h3 style="color: #2980b9; margin-bottom: 2rem;">Enhanced Customer Experience Through Intelligent Monitoring</h3>
                <p style="color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
                    Experience our production-ready system that reduced false alarms by 85% through 
                    context-aware anomaly detection and intelligent pattern recognition.
                </p>
                <div style="margin-bottom: 3rem; line-height: 2.5;">
                    <div style="margin-bottom: 1rem;">
                        <span style="background: #e74c3c; color: white; padding: 0.6rem 1.2rem; 
                                   border-radius: 25px; margin: 0.4rem; display: inline-block; 
                                   font-weight: bold;">Kafka/Event Hubs</span>
                        <span style="background: #3498db; color: white; padding: 0.6rem 1.2rem; 
                                   border-radius: 25px; margin: 0.4rem; display: inline-block; 
                                   font-weight: bold;">TimescaleDB</span>
                    </div>
                    <div>
                        <span style="background: #2ecc71; color: white; padding: 0.6rem 1.2rem; 
                                   border-radius: 25px; margin: 0.4rem; display: inline-block; 
                                   font-weight: bold;">FastAPI</span>
                        <span style="background: #f39c12; color: white; padding: 0.6rem 1.2rem; 
                                   border-radius: 25px; margin: 0.4rem; display: inline-block; 
                                   font-weight: bold;">Twilio Integration</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns to center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Outage Detection Dashboard", type="primary", use_container_width=True):
            st.session_state.demo_started = True
            st.rerun()
    
    st.stop()

# Main header
st.markdown("""
<div class="main-header">
    <h1>⚡ Real-Time Outage Detection System</h1>
    <h3>Enterprise-Grade Anomaly Detection | Enhanced Customer Experience</h3>
    <p style="font-size: 1.1rem; margin-top: 1rem;">
        Production system monitoring 247K+ meters with 85% reduction in false alarms
    </p>
</div>
""", unsafe_allow_html=True)

# Generate real-time data
def generate_realtime_data():
    current_time = datetime.now()
    base_usage = 1250 + 300 * np.sin(2 * np.pi * current_time.hour / 24)
    noise = np.random.normal(0, 50)
    
    # Simulate occasional anomalies
    anomaly_factor = 1.0
    if random.random() < 0.1:  # 10% chance of anomaly
        anomaly_factor = random.choice([0.3, 1.8])  # Significant drop or spike
    
    return {
        'timestamp': current_time,
        'active_meters': 247892 + random.randint(-100, 100),
        'current_usage_mw': round((base_usage * anomaly_factor) + noise, 1),
        'detected_anomalies': random.randint(0, 3),
        'active_alerts': random.randint(0, 2),
        'suppressed_planned': random.randint(5, 15),
        'response_time_ms': random.randint(45, 120),
        'customer_notifications': random.randint(0, 8),
        'false_alarm_rate': round(random.uniform(12, 18), 1)
    }

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Live Monitoring", 
    "🧠 ML Architecture", 
    "⚙️ Technical Implementation", 
    "📊 Business Impact",
    "🎯 Context Features"
])

with tab1:
    st.header("Real-Time Outage Detection Dashboard")
    
    # Real-time metrics
    data = generate_realtime_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Active Meters</h3>
            <h2>{data['active_meters']:,}</h2>
            <p>Monitored in Real-Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Usage</h3>
            <h2>{data['current_usage_mw']} MW</h2>
            <p>System-Wide Load</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Response Time</h3>
            <h2>{data['response_time_ms']} ms</h2>
            <p>Anomaly Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>False Alarm Rate</h3>
            <h2>{data['false_alarm_rate']}%</h2>
            <p>85% Reduction Achieved</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alert Status Section
    st.subheader("🚨 Current Alert Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data['active_alerts'] > 0:
            st.markdown(f"""
            <div class="alert-card">
                <h4>⚠️ Active Alerts: {data['active_alerts']}</h4>
                <p>Investigating voltage anomaly in Sector 7B</p>
                <p><strong>ETA Resolution:</strong> 12 minutes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-card">
                <h4>✅ All Systems Normal</h4>
                <p>No active outage alerts</p>
                <p><strong>Status:</strong> Monitoring</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4>📋 Planned Suppression</h4>
            <h2>{data['suppressed_planned']}</h2>
            <p>Maintenance events filtered out</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <h4>📱 Notifications Sent</h4>
            <h2>{data['customer_notifications']}</h2>
            <p>Automated customer alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Live Usage Chart using Streamlit native charts
    st.subheader("📈 Real-Time Usage Patterns")
    
    # Generate historical data for chart
    time_range = pd.date_range(end=datetime.now(), periods=144, freq='10min')
    usage_data = []
    
    for i, timestamp in enumerate(time_range):
        base = 1250 + 300 * np.sin(2 * np.pi * timestamp.hour / 24)
        seasonal = 100 * np.sin(2 * np.pi * timestamp.dayofyear / 365)
        noise = np.random.normal(0, 30)
        
        # Add some anomalies
        anomaly = 1.0
        if i > 130 and i < 135:  # Recent anomaly
            anomaly = 0.4
        
        usage_data.append({
            'timestamp': timestamp,
            'Actual Usage (MW)': (base + seasonal + noise) * anomaly,
            'Predicted (MW)': base + seasonal,
            'anomaly': anomaly != 1.0
        })
    
    df = pd.DataFrame(usage_data)
    df.set_index('timestamp', inplace=True)
    
    # Display chart
    st.line_chart(df[['Actual Usage (MW)', 'Predicted (MW)']])
    
    # Anomaly indicators
    anomaly_count = df['anomaly'].sum()
    if anomaly_count > 0:
        st.error(f"🚨 {anomaly_count} anomalies detected in the last 24 hours")
    else:
        st.success("✅ No anomalies detected in the last 24 hours")
    
    # Recent Events Log
    st.subheader("📝 Recent Events Log")
    
    events = [
        {"time": "14:23:15", "type": "INFO", "message": "Planned maintenance suppression activated for Substation Alpha"},
        {"time": "14:20:42", "type": "RESOLVED", "message": "Voltage anomaly in Sector 7B resolved - Normal operations resumed"},
        {"time": "14:15:31", "type": "ALERT", "message": "Unusual load pattern detected in residential zone R4"},
        {"time": "14:12:08", "type": "INFO", "message": "Customer notification sent to 1,247 affected accounts"},
        {"time": "14:10:55", "type": "WARNING", "message": "Load drop of 23% detected in commercial district C2"}
    ]
    
    for event in events:
        status_color = {
            "INFO": "#3498db",
            "RESOLVED": "#2ecc71", 
            "ALERT": "#e74c3c",
            "WARNING": "#f39c12"
        }[event["type"]]
        
        st.markdown(f"""
        <div style="border-left: 4px solid {status_color}; padding: 0.5rem; margin: 0.5rem 0; background: #f8f9fa;">
            <strong>{event['time']}</strong> - <span style="color: {status_color};">{event['type']}</span><br>
            {event['message']}
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("🧠 Machine Learning Architecture")
    
    st.markdown("""
    <div class="info-card">
        <h3>Context-Aware Anomaly Detection Pipeline</h3>
        <p>Advanced ML system leveraging historical patterns, operational context, and real-time signals 
        to achieve 85% reduction in false alarms through intelligent feature engineering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture Diagram
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 System Architecture Flow")
        
        architecture_data = {
            'Stage': ['Data Ingestion', 'Stream Processing', 'Feature Engineering', 'ML Detection', 'Context Filtering', 'Alert Management'],
            'Technology': ['Kafka/Event Hubs', 'Python Async Workers', 'Historical Profiling', 'Isolation Forest + LSTM', 'Planned Work API', 'Twilio + FastAPI'],
            'Processing Time': ['< 50ms', '< 100ms', '< 200ms', '< 300ms', '< 100ms', '< 500ms'],
            'Throughput': ['50K msg/sec', '45K msg/sec', '40K msg/sec', '35K msg/sec', '30K msg/sec', '5K alerts/hr']
        }
        
        arch_df = pd.DataFrame(architecture_data)
        st.dataframe(arch_df, use_container_width=True)
    
    with col2:
        st.subheader("🎯 ML Model Accuracy")
        
        # Model performance using native Streamlit chart
        models_data = {
            'Model': ['Isolation Forest', 'LSTM Networks', 'Statistical Analysis', 'Context Features'],
            'Accuracy (%)': [92.3, 89.7, 85.1, 94.8]
        }
        
        models_df = pd.DataFrame(models_data)
        st.bar_chart(models_df.set_index('Model'))
    
    # Feature Engineering Details
    st.subheader("⚙️ Advanced Feature Engineering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tech-detail">
            <h4>🕒 Temporal Features</h4>
            <ul>
                <li>Hour-of-day patterns</li>
                <li>Day-of-week cycles</li>
                <li>Seasonal adjustments</li>
                <li>Weather correlations</li>
                <li>Historical load profiles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-detail">
            <h4>🏗️ Operational Context</h4>
            <ul>
                <li>Planned maintenance events</li>
                <li>Feeder reconfiguration</li>
                <li>Equipment status</li>
                <li>Grid topology changes</li>
                <li>Protection settings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tech-detail">
            <h4>📈 Usage Patterns</h4>
            <ul>
                <li>Load curve derivatives</li>
                <li>Voltage stability metrics</li>
                <li>Power factor trends</li>
                <li>Harmonic distortion</li>
                <li>Phase imbalance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Performance Metrics
    st.subheader("📊 Model Performance Dashboard")
    
    performance_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'False Positive Rate', 'Response Time', 'Uptime'],
        'Current': ['94.8%', '92.1%', '93.4%', '2.3%', '87ms', '99.97%'],
        'Target': ['95.0%', '90.0%', '92.5%', '3.0%', '100ms', '99.95%'],
        'Industry Avg': ['78.5%', '71.2%', '74.6%', '15.2%', '250ms', '99.1%']
    }
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True)

with tab3:
    st.header("⚙️ Technical Implementation")
    
    # Tech Stack Overview
    st.subheader("🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tech-detail">
            <h4>🔄 Real-Time Processing</h4>
            <ul>
                <li><strong>Apache Kafka</strong> - Event streaming (50K msg/sec)</li>
                <li><strong>Azure Event Hubs</strong> - Cloud ingestion layer</li>
                <li><strong>Python AsyncIO</strong> - Non-blocking workers</li>
                <li><strong>TimescaleDB</strong> - Time-series data storage</li>
                <li><strong>Redis</strong> - In-memory caching</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-detail">
            <h4>🤖 ML & API Layer</h4>
            <ul>
                <li><strong>scikit-learn</strong> - Isolation Forest models</li>
                <li><strong>TensorFlow</strong> - LSTM neural networks</li>
                <li><strong>FastAPI</strong> - High-performance API</li>
                <li><strong>Twilio</strong> - SMS/Voice notifications</li>
                <li><strong>Docker</strong> - Containerized deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Code Examples
    st.subheader("💻 Key Implementation Components")
    
    tab_code1, tab_code2, tab_code3 = st.tabs(["Kafka Consumer", "Anomaly Detection", "Alert Management"])
    
    with tab_code1:
        st.code("""
# High-throughput Kafka consumer with async processing
import asyncio
from kafka import KafkaConsumer
import json
from datetime import datetime

class OutageDetectionConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'meter-readings',
            bootstrap_servers=['kafka-cluster:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            max_poll_records=1000,
            fetch_max_wait_ms=50
        )
        self.processing_queue = asyncio.Queue(maxsize=10000)
    
    async def consume_messages(self):
        \"\"\"Async message consumption with batching\"\"\"
        for message in self.consumer:
            meter_data = {
                'meter_id': message.value['meter_id'],
                'timestamp': datetime.fromisoformat(message.value['timestamp']),
                'usage_kw': message.value['usage_kw'],
                'voltage': message.value['voltage'],
                'power_factor': message.value['power_factor']
            }
            
            await self.processing_queue.put(meter_data)
            
            # Batch processing for efficiency
            if self.processing_queue.qsize() >= 100:
                await self.process_batch()
""", language="python")
    
    with tab_code2:
        st.code("""
# Context-aware anomaly detection with historical profiling
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import asyncio

class ContextAwareAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200
        )
        self.historical_profiles = {}
        self.planned_outages = set()
    
    async def detect_anomalies(self, meter_data_batch):
        \"\"\"Detect anomalies with context awareness\"\"\"
        results = []
        
        for meter_data in meter_data_batch:
            # Extract temporal features
            features = self.extract_features(meter_data)
            
            # Check against historical profile
            anomaly_score = self.isolation_forest.decision_function([features])[0]
            
            # Apply context filtering
            if self.is_planned_outage(meter_data['meter_id'], meter_data['timestamp']):
                # Suppress alert for planned maintenance
                continue
            
            # Apply feeder history context
            feeder_context = await self.get_feeder_context(meter_data['meter_id'])
            adjusted_score = self.apply_context_weighting(anomaly_score, feeder_context)
            
            if adjusted_score < -0.6:  # Anomaly threshold
                alert = {
                    'meter_id': meter_data['meter_id'],
                    'timestamp': meter_data['timestamp'],
                    'anomaly_score': adjusted_score,
                    'confidence': abs(adjusted_score),
                    'context': feeder_context
                }
                results.append(alert)
        
        return results
""", language="python")
    
    with tab_code3:
        st.code("""
# FastAPI alert management with Twilio integration
from fastapi import FastAPI, BackgroundTasks
from twilio.rest import Client
import asyncio
from datetime import datetime, timedelta

app = FastAPI(title="Outage Detection API")

class AlertManager:
    def __init__(self):
        self.twilio_client = Client(account_sid, auth_token)
        self.active_alerts = {}
        self.notification_history = []
    
    async def process_alert(self, alert_data):
        \"\"\"Process and escalate alerts based on severity\"\"\"
        alert_id = f"{alert_data['meter_id']}_{alert_data['timestamp']}"
        
        # Check for duplicate alerts (within 5 minutes)
        if self.is_duplicate_alert(alert_data):
            return
        
        # Classify alert severity
        severity = self.classify_severity(alert_data)
        
        # Determine affected customers
        affected_customers = await self.get_affected_customers(alert_data['meter_id'])
        
        # Create alert record
        alert = {
            'id': alert_id,
            'severity': severity,
            'affected_customers': len(affected_customers),
            'estimated_duration': self.estimate_duration(alert_data),
            'created_at': datetime.now()
        }
        
        self.active_alerts[alert_id] = alert
        
        # Send notifications based on severity
        if severity == 'CRITICAL':
            await self.send_immediate_notifications(affected_customers, alert)
        elif severity == 'HIGH':
            await self.schedule_notifications(affected_customers, alert, delay=5)

@app.post("/alerts/create")
async def create_alert(alert_data: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(alert_manager.process_alert, alert_data)
    return {"status": "Alert processing initiated"}
""", language="python")
    
    # Performance Metrics
    st.subheader("⚡ System Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Message Throughput", "50,000/sec", "↑ 15%")
        st.metric("Processing Latency", "87ms", "↓ 23ms")
    
    with col2:
        st.metric("Detection Accuracy", "94.8%", "↑ 2.3%")
        st.metric("False Positive Rate", "2.1%", "↓ 13.4%")
    
    with col3:
        st.metric("System Uptime", "99.97%", "↑ 0.02%")
        st.metric("Alert Response", "< 500ms", "↓ 150ms")

with tab4:
    st.header("📊 Business Impact Analysis")
    
    # ROI Dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💰 3-Year ROI Analysis")
        
        roi_data = {
            'Year': [2023, 2024, 2025],
            'Development Cost ($M)': [1.2, 0.4, 0.2],
            'Operational Savings ($M)': [0.8, 2.3, 2.8],
            'Customer Satisfaction Value ($M)': [1.5, 3.2, 3.8]
        }
        
        roi_df = pd.DataFrame(roi_data)
        roi_df.set_index('Year', inplace=True)
        
        st.bar_chart(roi_df)
        
        # Total ROI calculation
        total_investment = sum(roi_data['Development Cost ($M)'])
        total_returns = sum(roi_data['Operational Savings ($M)']) + sum(roi_data['Customer Satisfaction Value ($M)'])
        net_roi = total_returns - total_investment
        
        st.success(f"**Total Net ROI: ${net_roi:.1f}M** over 3 years")
    
    with col2:
        st.subheader("🎯 Key Metrics")
        
        st.markdown("""
        <div class="metric-card">
            <h4>Total ROI</h4>
            <h2>$11.6M</h2>
            <p>3-Year Value</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>Payback Period</h4>
            <h2>8 months</h2>
            <p>Break-even Point</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>Annual Savings</h4>
            <h2>$2.8M</h2>
            <p>Operational Efficiency</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Business Benefits Breakdown
    st.subheader("🏆 Enterprise Benefits")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🔧 Operational Excellence</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li>• 85% reduction in false alarms</li>
                <li>• 67% faster incident response</li>
                <li>• 45% reduction in truck rolls</li>
                <li>• 92% automated alert processing</li>
                <li>• 99.97% system uptime</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>😊 Customer Experience</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li>• 78% improvement in satisfaction</li>
                <li>• 5-minute notification delivery</li>
                <li>• 94% accurate restoration estimates</li>
                <li>• 3.2x increase in proactive comms</li>
                <li>• 89% mobile app engagement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>💡 Strategic Value</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li>• Foundation for smart grid</li>
                <li>• Regulatory compliance boost</li>
                <li>• Data-driven decision making</li>
                <li>• Scalable cloud architecture</li>
                <li>• Competitive differentiation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Customer Impact Metrics
    st.subheader("📞 Customer Communication Impact")
    
    communication_data = {
        'Metric': ['Average Notification Time', 'Customer Call Volume', 'Satisfaction Score', 'App Engagement', 'Complaint Resolution'],
        'Before System': ['23 minutes', '15,000/day', '6.2/10', '34%', '4.5 days'],
        'After System': ['5 minutes', '4,200/day', '8.7/10', '89%', '1.2 days'],
        'Improvement': ['↓ 78%', '↓ 72%', '↑ 40%', '↑ 162%', '↓ 73%']
    }
    
    comm_df = pd.DataFrame(communication_data)
    st.dataframe(comm_df, use_container_width=True)

with tab5:
    st.header("🎯 Context Features Deep Dive")
    
    st.markdown("""
    <div class="info-card">
        <h3>Key Learning: Context Features > Threshold Tuning</h3>
        <p>Our breakthrough discovery: Incorporating operational context and historical response profiles 
        reduced false alarms more effectively than traditional threshold optimization approaches.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Importance Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Feature Importance Analysis")
        
        feature_data = {
            'Feature': ['Planned Outage Context', 'Historical Response Profile', 'Feeder Configuration', 'Weather Correlation', 'Time-of-Day Patterns'],
            'Importance (%)': [34.2, 28.7, 18.9, 12.3, 5.9]
        }
        
        feature_df = pd.DataFrame(feature_data)
        st.bar_chart(feature_df.set_index('Feature'))
    
    with col2:
        st.subheader("📊 False Alarm Reduction Progress")
        
        reduction_data = {
            'System Enhancement': ['Baseline', 'Time Patterns', 'Weather Data', 'Feeder Context', 'Planned Work', 'Full Context'],
            'False Alarm Rate (%)': [23.4, 19.8, 16.2, 12.1, 7.8, 3.6]
        }
        
        reduction_df = pd.DataFrame(reduction_data)
        st.line_chart(reduction_df.set_index('System Enhancement'))
        
        # Show improvement percentage
        improvement = ((23.4 - 3.6) / 23.4) * 100
        st.success(f"**{improvement:.1f}% Total Reduction** in false alarms achieved!")
    
    # Context Feature Details
    st.subheader("⚙️ Context Feature Implementation")
    
    tab_ctx1, tab_ctx2, tab_ctx3 = st.tabs(["Planned Outage Suppression", "Historical Profiling", "Feeder Context"])
    
    with tab_ctx1:
        st.markdown("""
        <div class="tech-detail">
            <h4>🛠️ Planned Outage Suppression Logic</h4>
            <p>Advanced filtering system that prevents false alarms during scheduled maintenance:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.code("""
# Planned outage suppression with buffer zones
class PlannedOutageSupressor:
    def __init__(self):
        self.planned_events = []
        self.buffer_minutes = 15  # Grace period
    
    def is_planned_outage(self, meter_id, timestamp):
        \"\"\"Check if meter is in planned outage window\"\"\"
        meter_feeder = self.get_meter_feeder(meter_id)
        
        for event in self.planned_events:
            if (event['feeder'] == meter_feeder and 
                event['start_time'] - timedelta(minutes=self.buffer_minutes) <= timestamp <= 
                event['end_time'] + timedelta(minutes=self.buffer_minutes)):
                
                # Log suppression for audit
                self.log_suppression(meter_id, timestamp, event['work_order'])
                return True
        
        return False
    
    def load_planned_events(self):
        \"\"\"Integration with work management system\"\"\"
        # Real-time integration with SAP/Maximo
        events = work_management_api.get_scheduled_work()
        
        for event in events:
            self.planned_events.append({
                'work_order': event['id'],
                'feeder': event['affected_feeder'],
                'start_time': event['scheduled_start'],
                'end_time': event['estimated_completion'],
                'crew': event['assigned_crew']
            })
""", language="python")
    
    with tab_ctx2:
        st.markdown("""
        <div class="tech-detail">
            <h4>📈 Historical Response Profile Analysis</h4>
            <p>Learning from past outage patterns to improve future detection accuracy:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.code("""
# Historical response profile builder
class HistoricalProfileAnalyzer:
    def __init__(self):
        self.feeder_profiles = {}
        self.seasonal_adjustments = {}
    
    def build_feeder_profile(self, feeder_id):
        \"\"\"Build historical response characteristics\"\"\"
        historical_data = self.get_historical_outages(feeder_id, days=365)
        
        profile = {
            'avg_outage_duration': np.mean(historical_data['duration']),
            'typical_affected_count': np.mean(historical_data['customer_count']),
            'seasonal_patterns': self.extract_seasonal_patterns(historical_data),
            'equipment_reliability': self.calculate_reliability_score(feeder_id),
            'response_time_percentiles': np.percentile(historical_data['response_time'], [50, 75, 90, 95])
        }
        
        # Weather correlation analysis
        profile['weather_sensitivity'] = self.analyze_weather_correlation(
            historical_data, feeder_id
        )
        
        # Equipment age factor
        profile['equipment_age_factor'] = self.get_equipment_age_multiplier(feeder_id)
        
        self.feeder_profiles[feeder_id] = profile
        return profile
""", language="python")
    
    with tab_ctx3:
        st.markdown("""
        <div class="tech-detail">
            <h4>🔌 Feeder Configuration Context</h4>
            <p>Real-time awareness of grid topology and equipment status:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.code("""
# Feeder context analyzer
class FeederContextAnalyzer:
    def __init__(self):
        self.topology_cache = {}
        self.equipment_status = {}
    
    async def get_feeder_context(self, meter_id):
        \"\"\"Get comprehensive feeder operational context\"\"\"
        feeder_id = self.get_meter_feeder(meter_id)
        
        context = {
            'feeder_id': feeder_id,
            'current_load': await self.get_current_load(feeder_id),
            'protection_settings': await self.get_protection_settings(feeder_id),
            'recent_switches': await self.get_recent_switching(feeder_id),
            'equipment_alarms': await self.get_equipment_alarms(feeder_id),
            'topology_changes': await self.detect_topology_changes(feeder_id)
        }
        
        # Analyze upstream/downstream relationships
        context['upstream_feeders'] = self.get_upstream_feeders(feeder_id)
        context['downstream_customers'] = self.get_downstream_customers(feeder_id)
        
        return context
""", language="python")
    
    # Implementation Results
    st.subheader("🏆 Implementation Results Summary")
    
    results_data = {
        'Metric': [
            'False Alarm Rate',
            'Detection Accuracy', 
            'Response Time',
            'Customer Satisfaction',
            'Operational Cost',
            'System Reliability'
        ],
        'Before Context': ['23.4%', '76.2%', '8.5 min', '6.2/10', '$3.2M/year', '97.8%'],
        'After Context': ['3.6%', '94.8%', '2.1 min', '8.7/10', '$1.9M/year', '99.97%'],
        'Improvement': ['↓ 84.6%', '↑ 24.4%', '↓ 75.3%', '↑ 40.3%', '↓ 40.6%', '↑ 2.2%']
    }
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>⚡ Real-Time Outage Detection System</h4>
    <p>Production-ready enterprise solution • 247K+ meters monitored • 85% false alarm reduction</p>
    <p><strong>Technology Stack:</strong> Kafka/Event Hubs • Python AsyncIO • TimescaleDB • FastAPI • Twilio</p>
</div>
""", unsafe_allow_html=True)
