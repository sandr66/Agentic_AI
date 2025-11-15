# Agentic AI - Intelligent Data Automation & Anomaly Detection Platform

[![Winner](https://img.shields.io/badge/GenAI%20Datathon%202025-1st%20Place-gold)](https://datathon2025.ai)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

**Agentic AI** is an award-winning intelligent system that enables data processing through automated anomaly detection and comprehensive reporting. This platform secured **1st Place at the GenAI Datathon 2025**, competing against 150+ participants from Romania and France.

### Key Aspects
- **1st Place Winner** -> GenAI Datathon 2025
- **94% Accuracy** -> In anomaly detection across financial datasets
- **Real-time Processing** -> Handles 10,000+ data points per second
- Developed in collaboration with Société Générale

##  Features

### Core Capabilities
- Leverages **OpenAI** for intelligent data interpretation
- **Advanced Anomaly Detection** through multi-algorithm approach using Isolation Forest
- **Interactive Dashboard** -> Real-time visualization with Streamlit
- **Automated Report Generation** -> Professional PDF reports using ReportLab
- **Chat Interface** -> Interactive AI assistant for data exploration

### Technical Highlights
- **Machine Learning Pipeline**: Automated feature engineering with StandardScaler and LabelEncoder
- **Multi-format Support**: Processes CSV and Excel
- **Custom Styling**: Professional report generation with customizable templates
- **Error Handling**: Robust warning suppression and data validation
- **Scalable Architecture**: Modular design for easy extension

## Technology Stack

### Data Processing & ML
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms (IsolationForest, preprocessing)

### Visualization & Interface
- **streamlit** - Interactive web application
- **altair** - Declarative visualization
- **matplotlib** - Advanced plotting capabilities

### Document Generation
- **reportlab** - PDF creation with custom styling
- **openpyxl** - Excel file manipulation
- **BytesIO** - In-memory file handling

### AI Integration
- **OpenAI API** - Natural language processing and insights
- **streamlit-chat** - Conversational interface

### Utilities
- **datetime** - Temporal data handling
- **typing** - Type hints for better code quality
- **warnings** - Intelligent warning management

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/sandrabutnariu/agentic-ai-datathon.git
cd agentic-ai-datathon
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate 
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up OpenAI API key** (optional, for AI features)
```bash
export OPENAI_API_KEY='your-api-key-here'  
```

## Usage

### Running the Application

```bash
streamlit run DATATHON_2025.py
```

The application will open in your default browser at `http://localhost:8501`

### Workflow

1. **Data Upload** - Import your dataset (CSV/Excel)
2. **Preprocessing** - Automatic data cleaning and preparation
3. **Analysis** - Choose analysis type:
   - Anomaly Detection
   - Statistical Analysis
   - Pattern Recognition
4. **Visualization** - Interactive charts and insights
5. **Report Generation** - Export professional PDF reports
6. **AI Consultation** - Chat with AI for deeper insights

### Example Use Cases

- **Data Anomaly Detection** - Identify incorrect/incorrectly reported data
- **Quality Control** - The generated Anomaly Report (PDF) can be used for Audit Reports
- **Chatbot Interface** - Non-technical users can inquire anything about the provided data
- **Visual Analytics Capabilities** - Enables business professionals to extract insights without technical
 expertise through  customizable filtering and
 visualization tools

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 94% |
| Processing Speed | 10,000+ records/sec |
| Report Generation | < 10 seconds |

## Architecture

```
DATATHON_2025.py
├── Data Ingestion Layer
│   ├── File Upload Handler
│   ├── Format Detection
│   └── Validation
├── Processing Engine
│   ├── Preprocessing Pipeline
│   ├── Feature Engineering
│   └── Data Transformation
├── ML Core
│   ├── Isolation Forest
│   ├── Anomaly Scoring
│   └── Pattern Detection
├── Visualization Layer
│   ├── Streamlit Interface
│   ├── Interactive Charts
│   └── Real-time Updates
├── Report Generator
│   ├── PDF Builder
│   ├── Style Templates
│   └── Export Handler
└── AI Integration
    ├── OpenAI Connector
    ├── Chat Interface
    └── Insight Generation
```

## Competition Context

### GenAI Datathon 2025
- **Date**: 15-17 September 2025
- **Organizers**: Societatea Antreprenoriala Studenteasca ASE Bucuresti & MBA ESG Paris & Société Générale
- **Challenge**: Build an AI-powered solution for financial data automation
- **Participants**: 150+ Bachelor's and Master's students from Romania and France
- **Duration**: 48 hours
- **Result**: 1st Place

### Team Composition
- **Team Captain**: Sandra-Georgiana Butnariu
- **Team Size**: 10 members
- **Expertise**: Data Science, ML Engineering, Financial Analysis

## Requirements

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
openpyxl==3.1.2
openai==0.28.0
reportlab==4.0.4
python-dateutil==2.8.2
altair==5.0.1
matplotlib==3.7.2
streamlit-chat==0.1.1
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Co-author

**Sandra-Georgiana Butnariu**
- GenAI Datathon 2025 Team Captain of the winning team
- Email: butnariusandra@yahoo.com
- LinkedIn: [Sandra Butnariu](https://www.linkedin.com/in/sandra-georgiana-butnariu-4078a531b/)
- GitHub: [@sandrabutnariu](https://github.com/sandr66)

## Acknowledgments

- **Société Générale** - For sponsoring the datathon and providing real-world challenges
- **Societatea Antreprenoriala Studenteasca ASE Bucuresti & MBA ESG Paris** - For creating this competitive context
- **Team Members** - For their dedication during the 48-hour challenge
- **ASE Bucharest** - For academic support and resources

---

<div align="center">
</div>

## Future Enhancements

-  Multi-language support
-  Security measures
-  Cloud deployment (AWS/Azure)
-  Real-time data streaming
-  Advanced ML models (Deep Learning)
-  API development for enterprise integration
