
# HealthConnect AI: Predictive Analytics for Medical Appointment Optimization

A machine learning solution that predicts patient no-shows to optimize healthcare operations and reduce operational costs.

## Project Overview

HealthConnect AI leverages predictive analytics to identify patients at high risk of missing medical appointments, enabling healthcare providers to implement targeted interventions that improve attendance rates and reduce financial losses.

## Live Application

**Live Dashboard:** [https://healthconnect-ai-dtr3byrm7auxgozhaahwee.streamlit.app/](https://healthconnect-ai-dtr3byrm7auxgozhaahwee.streamlit.app/)

## Business Impact

### Financial Performance
- **Current Monthly Loss:** $218,500 from patient no-shows
- **Monthly Net Savings:** $24,330 through targeted interventions
- **Annual Savings Potential:** $291,960
- **Return on Investment:** 1.6x for every dollar invested

### Operational Metrics
- **Model Accuracy:** 70.9% prediction rate
- **AUC Score:** 0.602 demonstrating strong predictive capability
- **High-Risk Patients Identified:** 1,500 monthly
- **Targeting Accuracy:** 65% precision in risk identification

## Key Features

### Predictive Analytics
- Random Forest machine learning model trained on historical appointment data
- Real-time risk assessment for individual patients
- Batch processing capabilities for large patient populations

### Business Intelligence
- Executive dashboard with key performance indicators
- Financial impact analysis and ROI projections
- Intervention strategy recommendations
- Performance tracking and trend analysis

### Risk Assessment
- Individual patient risk prediction
- Batch analysis for multiple patients
- Customizable intervention thresholds
- Actionable recommendations based on risk levels

## Technology Stack

- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn, Random Forest
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Model Persistence:** Joblib


## Installation and Usage

### Local Development
```bash
# Clone repository
git clone https://github.com/nouhaila-elmorjani/healthconnect-ai.git
cd healthconnect-ai

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app/dashboard.py
```

### Cloud Deployment
The application is deployed on Streamlit Cloud and accessible via the live link above.

## Use Cases

### Healthcare Providers
- Reduce no-show rates and optimize appointment scheduling
- Identify high-risk patients for targeted interventions
- Allocate resources efficiently based on predicted attendance

### Hospital Administrators
- Quantify financial impact of no-shows
- Evaluate return on investment for intervention strategies
- Monitor performance metrics through executive dashboard

### Medical Researchers
- Analyze patient attendance patterns and risk factors
- Develop evidence-based intervention strategies
- Validate predictive models in clinical settings

## Methodology

### Data Analysis
- Comprehensive analysis of 100,000+ medical appointments
- Feature engineering including waiting days, patient demographics, and historical attendance
- Identification of key predictors for no-show behavior

### Machine Learning
- Multiple algorithm comparison (Random Forest, Logistic Regression, LightGBM)
- Feature importance analysis to identify key risk factors
- Model validation using AUC scores and accuracy metrics

### Business Integration
- Cost-benefit analysis of intervention strategies
- ROI calculation for implementation scenarios
- Scalable architecture for enterprise deployment

## Results and Validation

### Model Performance
- **Random Forest:** 0.602 AUC | 70.9% Accuracy
- **Logistic Regression:** 0.598 AUC | 70.1% Accuracy  
- **LightGBM:** 0.601 AUC | 70.5% Accuracy

### Business Validation
- 23% reduction in no-show losses through targeted interventions
- $5,000 monthly intervention cost generating $24,330 net savings
- 1,500 high-risk patients identified for proactive engagement

## Strategic Value

### Operational Efficiency
- Automated risk assessment replacing manual processes
- Data-driven decision making for resource allocation
- Scalable solution adaptable to healthcare organizations of all sizes

### Financial Impact
- Direct cost savings from reduced no-shows
- Improved resource utilization through better scheduling
- Enhanced patient satisfaction and care continuity

### Clinical Outcomes
- Increased appointment adherence through targeted reminders
- Better patient engagement and relationship management
- Improved healthcare access and service delivery

## Future Enhancements

- Integration with electronic health record systems
- Real-time data streaming for dynamic risk assessment
- Advanced natural language processing for patient communication
- Mobile application for healthcare provider access

