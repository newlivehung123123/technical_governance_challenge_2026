# technical_governance_challenge_2026
This repository hosts the outputs of the exploratory stage of Phases 3-4 of my GAID Project, featuring an automated global audit of geographical bias in the open-weight Llama-3 8B model. This exploratory project evaluates 1,704 queries across 213 countries to expose knowledge gaps in AI safety, fairness and readiness within the Global South.

# 🌍 Global AI Bias Audit for Technical Governance

This repository contains the code, data, and technical findings from the exploratory phase of the [Global AI Dataset (GAID) Project](https://dataverse.harvard.edu/dataverse/gaidproject). This exploratory research evaluates geographical and socioeconomic biases in Large Language Models (LLMs), specifically stress-testing the Llama-3 8B model's awareness of technical AI governance metrics across 213 countries.

## 📑 Project Overview
Current AI alignment processes often reinforce geoeconomic asymmetries, potentially leading to a modern form of **digital colonization**. This project provides a quantitative, data-driven audit to expose "geographical hallucinations" and information gaps between the Global North and South.

## 🚀 Quickstart

### Prerequisites
* **Hardware:** NVIDIA A100 GPU (recommended) or T4 (via Google Colab).
* **Dependencies:** `transformers`, `bitsandbytes`, `accelerate`, `flash-attn`.

### Setup & Execution
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/newlivehung123123/technical_governance_challenge_2026.git](https://github.com/newlivehung123123/technical_governance_challenge_2026.git)
   cd technical_governance_challenge_2026

2. Run Inference: Open `technical_governance.ipynb` in Google Colab. This notebook handles the 4-bit NF4 quantization and Flash Attention 2 environment required to run the audit at scale.

3. Analyze Results:

    ```Bash
    python colab_analyze_audit_results.py
    This script processes the raw output to generate the statistical breakdown of "Knowledge Rate" vs. "Refusal Rate."

🛠️ Technical Methodology
Architecture
   ```text
   GAID Dataset (Ground Truth) 
 -> Metric Operationalization (8 Metrics)
 -> Automated Prompt Engineering (1,704 Queries)
 -> Model Audit (Llama-3 8B @ 4-bit Quantization)
 -> Response Categorization (Regex-based Pattern Matching)
 -> Statistical Analysis & Visualisation (Matplotlib/Seaborn)
```
  
### 🔍 Key Findings
* **📉 Low Factual Accuracy:** The model provided number/fact responses in only 11.4% of its query answers.
* **🚫 Systemic Ignorance:** High refusal rates were heavily concentrated in Sub-Saharan Africa and Latin America.
* **⚠️ Epistemic Exclusion:** The model frequently failed to recognize the technical agency and infrastructure of developing nations.
* **💰 Wealth Bias:** AI technical knowledge is heavily concentrated in higher-income regions.

## 📁 Repository Structure
* **📄 `technical paper.pdf`**: The full technical report presented at Apart Research's Technical AI Governance Challenge 2026.
* **📊 `census_audit_results.csv`**: The raw output of 1,704 unique audit queries across 8 technical metrics and 213 countries.
* **🐍 `colab_analyze_audit_results.py`**: Python script used to process model performance and calculate ignorance rates.

## 🛠️ Metrics Evaluated
The audit stress-tests three pillars of technical governance:
1. **🛡️ AI Safety:** Total training compute, hardware frontiers, and high-level publications.
2. **⚖️ AI Fairness:** Private investment, AI patents granted, and workforce size.
3. **🚀 AI Readiness:** Government AI readiness index and specialized infrastructure scores.

## 🏁 Getting Started
1. **View the Results:** Explore `census_audit_results.csv` to see how Llama-3 8B responded to specific jurisdictions.
2. **Run the Analysis:** Use the provided `.py` script to replicate the "Knowledge Rate" vs. "Refusal Rate" findings.
3. **Explore the Dataset:** The ground-truth data is hosted at the [GAID Project Dataverse](https://dataverse.harvard.edu/dataverse/gaidproject). **Note:** This exploratory project uses the GAID dataset as a framework to run AI evals.

## ✉️ Contact & Citation
**Author:** Jason Hung
**Affiliation:** Independent / Apart Research

If you use this data or code, please cite the original work:
> Hung, J. (2026). Global AI Bias Audit for Technical Governance. The Technical AI Governance Challenge 2026. Organised by Apart Research. https://doi.org/10.48550/arXiv.2602.13246
