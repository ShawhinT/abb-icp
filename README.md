# Ideal Customer Profile (ICP) Analysis

This repository contains code for creating an Ideal Customer Profile (ICP) for my AI Builders Bootcamp. It uses real student data (e.g. job title, company, join reason) to predict the user engagement and course KPIs. The results are then synthesized into a [final ICP](https://github.com/ShawhinT/abb-icp/blob/main/data/4-icp/icp.md), which can be repurposed for sales and marketing.

**Resources**
- [YouTube Video]

## Steps

1. **Data Preparation**: Combines and cleans student data from multiple cohorts
2. **Feature Engineering**: Creates meaningful features from raw data using AI classification
3. **Predictive Modeling**: Uses logistic regression to identify factors that predict high ratings
4. **Predictor Analysis**: Interpreting top variables from logistic regression model and qualitative analysis with GPT-4.1
5. **ICP Generation**: Creates a detailed ideal customer profile using GPT-4.1

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- uv (Python package manager)
- OpenAI API key

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd abb-icp
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up your OpenAI API key:
- Create a `.env` file and add your OpenAI API key
```
OPENAI_API_KEY=sk-proj-XXXX
```

## Project Structure
```
abb-icp/
├── data/
│ ├── 1-raw/ # Original data files
│ ├── 2-clean/ # Processed data
│ ├── 3-results/ # Model results and visualizations
│ └── 4-icp/ # Final ICP documents
├── utils/ # Helper functions
├── prompts/ # AI prompt templates
├── 1-data_prep.ipynb # Data preparation notebook
├── 2-feature_engineering.ipynb # Feature engineering notebook
├── 3-icp_analysis.ipynb # ICP analysis notebook
├── 4-predictor_analysis.ipynb # Predictor analysis notebook
└── 5-create_icp.ipynb # ICP generation notebook

```

## Results

The final ICP is saved in `data/4-icp/icp.md` and includes:
- Detailed customer demographics and psychographics
- Pain points and motivations
- Clear definition of who is NOT the ideal customer
- Actionable insights for marketing and sales teams

## License

This project is for educational purposes. Please ensure you have proper permissions for any data used in your own analysis.
