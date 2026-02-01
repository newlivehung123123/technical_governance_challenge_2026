# technical_governance_challenge_2026
This repository hosts the outputs of the exploratory stage of Phases 3-4 of my GAID Project, featuring an automated global audit of geographical bias in the open-weight Llama-3 8B model. This exploratory project evaluates 1,704 queries across 213 countries to expose knowledge gaps in AI safety, fairness and readiness within the Global South.

# 🌍 Global AI Bias Audit for Technical Governance

[cite_start]This repository contains the code, data, and technical report (including empirical findings) from the exploratory phase of the **Global AI Dataset (GAID) Project**[cite: 5, 6]. [cite_start]This research evaluates geographical and socioeconomic biases in Large Language Models (LLMs), specifically stress-testing the Llama-3 8B model's awareness of technical AI governance metrics across 213 countries[cite: 6, 7].

## 📑 Project Overview
[cite_start]Current AI alignment processes often reinforce geoeconomic asymmetries[cite: 11]. [cite_start]This project provides a quantitative, data-driven audit to expose "geographical hallucinations" and information gaps between the Global North and South[cite: 10, 18].

### Key Findings
* [cite_start]**Low Factual Accuracy:** The model provided number/fact responses in only 11.4% of queries[cite: 8].
* [cite_start]**Systemic Ignorance:** High refusal rates were concentrated in Sub-Saharan Africa and Latin America & the Caribbean [cite: 9, 135].
* [cite_start]**Epistemic Exclusion:** The model frequently failed to recognize the technical agency and infrastructure of developing countries[cite: 35, 36].

## 📁 Repository Structure
* [cite_start]**`GAID_Technical_Paper.pdf`**: The full technical report (in arXiv) presented at Apart Research's Technical AI Governance Challenge 2026[cite: 1, 12].
* [cite_start]**`census_audit_results.csv`**: The raw output of 1,704 queries across 8 technical metrics and 213 countries[cite: 7, 85].
* [cite_start]**`colab_analyze_audit_results.py`**: Python script used to categorize model responses (e.g., "Honest Ignorance" vs. "Unverified Confidence") and generate statistical insights[cite: 80, 91, 95].
* [cite_start]**`scripts/`**: Includes the automated query generation tools used in the audit[cite: 80].

## 🛠️ Metrics Evaluated
[cite_start]The audit stress-tests three pillars of technical governance[cite: 30, 31]:
1. [cite_start]**AI Safety:** Training compute, hardware frontiers, and high-level publications[cite: 71].
2. [cite_start]**AI Fairness:** Private investment, patents granted, and workforce size[cite: 72].
3. [cite_start]**AI Readiness:** Government readiness indices and specialized infrastructure[cite: 73].

## 🚀 Getting Started
1. **View the Results:** Explore `census_audit_results.csv` to see how Llama-3 8B responded to specific countries.
2. [cite_start]**Run the Analysis:** Use the provided `.py` script to replicate the "Knowledge Rate" vs. "Refusal Rate" findings[cite: 88, 182].
3. [cite_start]**Explore the Dataset:** The ground-truth global panel AI data is hosted at the [GAID Project Dataverse](https://doi.org/10.7910/DVN/PUMGYU)[cite: 19, 20].

## ✉️ Contact & Citation
[cite_start]**Author:** Jason Hung [cite: 2]
[cite_start]**Affiliation:** Independent / Apart Research [cite: 2, 3]

If you use this data or code, please cite the original arXiv paper:
> Hung, J. (2026). *Global AI Bias Audit for Technical Governance*. GAID Project.
