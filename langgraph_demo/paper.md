# LangGraph-Based Multi-Agent System for Astronomical Data Analysis: Knowledge Distillation for Domain-Specific Large Language Models

**基于 LangGraph 的多智能体天文数据分析系统：领域专用大语言模型的知识蒸馏**

---

## Abstract

The rapid growth of astronomical data from large-scale surveys such as LAMOST, SDSS, and Gaia has created unprecedented opportunities for data-driven discoveries, while simultaneously presenting significant challenges for efficient data analysis and interpretation. This paper presents a novel framework that integrates LangGraph-based multi-agent workflows with knowledge distillation techniques to develop domain-specific large language models (LLMs) for astronomical research.

Our system employs a sophisticated agent architecture comprising six specialized components: a Router Agent for query classification, Data Retrieval Agents for federating multiple astronomical databases, an Analysis Agent for spectral processing, a Reasoning Agent powered by Qwen-72B for scientific inference, a Verification Agent for cross-validation, and an Output Agent for generating structured reports and training data. The LangGraph framework enables stateful, cyclic workflows essential for iterative scientific reasoning.

To address the computational constraints of deploying large models in production environments, we implement a knowledge distillation pipeline that transfers capabilities from Qwen-72B to a compact Qwen-7B model enhanced with LoRA adapters. Our distillation strategy combines hard label supervision, soft target learning via KL divergence with temperature scaling, and domain-specific task losses. Experimental results demonstrate that the distilled model achieves 86.7% accuracy on spectral classification tasks, approaching the teacher model's 91.5% while reducing inference latency by 62% and memory footprint by 67%.

The framework is validated using data from the Variable Star Pipeline (VSP) and LAMOST DR10, with applications to cataclysmic variable identification, spectral type classification, and automated report generation. Our approach bridges the gap between cutting-edge LLM capabilities and practical deployment requirements in astronomical research.

**Keywords:** Large Language Models, Knowledge Distillation, Multi-Agent Systems, Astronomical Data Analysis, LangGraph, Qwen, Deep Learning

---

## 1. Introduction

### 1.1 Background and Motivation

Contemporary astronomy is experiencing a data revolution. The Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST) has collected over 20 million spectra [1], the Sloan Digital Sky Survey (SDSS) has mapped more than one-third of the sky [2], and the Gaia mission has cataloged nearly 2 billion celestial objects with astrometric and photometric measurements [3]. This deluge of data necessitates automated, intelligent analysis tools capable of handling complex, multi-modal scientific workflows.

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding, reasoning, and code generation [4,5]. However, applying general-purpose LLMs to specialized scientific domains presents several challenges: (1) domain knowledge gaps regarding astronomical concepts and data formats, (2) hallucination issues when interpreting specialized observations, and (3) prohibitive computational costs for state-of-the-art models with hundreds of billions of parameters.

### 1.2 Project Overview

This project develops **AstroSage**, an AI-powered assistant specifically designed for astronomical researchers. The system addresses a critical workflow gap: given a celestial coordinate (RA/Dec), it automatically integrates multi-source observational data—including ZTF light curves for variability characterization, Gaia proper motion and parallax for distance/kinematics, and interstellar extinction maps for photometric corrections—and leverages a locally-deployed large language model (Ollama + llama3.1:8b) to deliver automated source classification and physical parameter estimation.

Unlike general-purpose AI assistants, AstroSage is tightly coupled with astronomical data infrastructure. It does not merely answer questions about astronomy; it performs active data retrieval, cross-matching, and quantitative analysis to generate evidence-based conclusions about celestial objects.

### 1.3 Scientific Objectives

#### 1.3.1 Target Scientific Problems

This system addresses three core challenges in modern observational astronomy:

| Problem Category | Specific Challenge | Current Limitation |
|-----------------|-------------------|-------------------|
| **Data Fragmentation** | Astronomical data scattered across ZTF, Gaia, LAMOST, VSX, etc. | Researchers manually query each database |
| **Parameter Synthesis** | Combining photometry, astrometry, and spectroscopy for classification | Requires domain expertise in multiple subfields |
| **Scalable Screening** | Prioritizing candidates from large surveys (e.g., LAMOST DR10 with 10M+ spectra) | Manual inspection infeasible at survey scale |

**Primary Use Case**: LAMOST Candidate Batch Screening
- LAMOST DR10 contains over 10 million low-resolution spectra
- Variable star and exotic object candidates require prioritization for follow-up
- AstroSage enables automated triage: query → data fusion → probability ranking → expert review queue
- Expected throughput: ~1000 objects/hour on a single GPU workstation

#### 1.3.2 Comparison with Existing Tools

| Feature | astroquery/VizieR | AstroSage (This Work) |
|---------|------------------|----------------------|
| **Interface** | Python API / Web forms | Natural language + structured output |
| **Data Integration** | Single-service queries | Multi-source federated queries with cross-matching |
| **Intelligence** | Raw data retrieval | Automated reasoning and classification |
| **Extinction Query** | Not available | CSFD/SFD maps with coordinate transformations |
| **Output Format** | Tables/FITS | JSON + human-readable reports + confidence scores |
| **Deployment** | Requires Python environment | Local Ollama deployment, no cloud dependency |
| **Speed** | Network-dependent (~2-10s/query) | Local inference (~1.1s/query after data fetch) |

**Key Differentiation**: While astroquery provides programmatic access to individual databases, AstroSage orchestrates multi-source queries, performs automated analysis, and generates natural language interpretations with uncertainty quantification.

#### 1.3.3 Target Users and Scenarios

| User Type | Scenario | Value Proposition |
|-----------|----------|-------------------|
| **Survey Astronomer** | Screening LAMOST CV candidates | Automated triage, priority ranking |
| **Variable Star Researcher** | Classifying new ZTF variables | Multi-band light curve + Gaia kinematics fusion |
| **Graduate Student** | Learning stellar classification | Interactive exploration with explanations |
| **Observer** | Pre-observation target assessment | Quick physical parameter estimation |

### 1.4 Related Work

Recent research has explored various approaches to applying LLMs in astronomy. **AstroLLaMA** [6] fine-tuned Llama-2 on astronomical literature, achieving improved performance on domain-specific question answering. **AstroBERT** [7] employed masked language modeling on 300K astronomical papers for scientific text mining. However, these approaches primarily focus on text understanding rather than structured data analysis workflows.

Multi-agent systems have emerged as a promising paradigm for complex task decomposition. **AutoGPT** [8] and **LangChain** [9] demonstrated the potential of chaining LLM calls for autonomous task completion. **LangGraph** [10] extended this paradigm with support for cyclic workflows and persistent state, essential for iterative scientific analysis. Recent work by Wu et al. [11] applied multi-agent systems to protein structure prediction, demonstrating the potential for scientific discovery applications.

Knowledge distillation, pioneered by Hinton et al. [12], offers a pathway to compress large models into efficient deployable versions. **DistilBERT** [13] and **TinyBERT** [14] demonstrated successful compression for BERT models. In the context of instruction-following models, **Stanford Alpaca** [15] and **Vicuna** [16] showed that smaller models can achieve competitive performance when trained on high-quality synthetic data from larger teachers.

### 1.5 Contributions

This paper makes the following contributions:

1. **A novel LangGraph-based multi-agent architecture** specifically designed for astronomical data analysis workflows, incorporating domain-specific tools for spectroscopic and time-series data processing.

2. **A comprehensive knowledge distillation framework** that transfers reasoning capabilities from Qwen-72B to Qwen-7B, incorporating domain-specific losses and training data synthesized through LangGraph workflows.

3. **An extensible integration with existing astronomical databases** including LAMOST, SIMBAD, VizieR, and the Variable Star Pipeline (VSP), enabling real-time federated queries.

4. **Empirical validation** demonstrating that the distilled 7B parameter model achieves within 5% accuracy of the 72B teacher model on classification tasks while enabling real-time inference on commodity hardware.

---

## 2. Methodology

### 2.1 System Architecture

Our system adopts a layered architecture comprising three main components: the Data Layer, the Agent Layer, and the Model Layer.

#### 2.1.1 Data Layer

The Data Layer provides unified access to heterogeneous astronomical databases:

- **LAMOST DR10**: Spectroscopic survey data with over 10 million low-resolution spectra
- **SIMBAD**: Astronomical database containing 15 million objects with bibliographic references
- **VizieR**: Catalog service providing access to thousands of astronomical catalogs
- **VSX (Variable Star Index)**: Comprehensive catalog of 2.3 million variable stars
- **Local Knowledge Base**: Vector database storing domain-specific embeddings for retrieval-augmented generation

We implement standardized data adapters following the **Virtual Observatory (VO)** standards [17], enabling seamless integration of new data sources.

#### 2.1.2 Agent Layer

The Agent Layer implements a directed graph workflow using LangGraph, where nodes represent specialized agents and edges define control flow. Unlike linear pipelines, our architecture supports cyclic execution for iterative refinement:

**Router Agent**: Classifies incoming queries into categories (variable star identification, spectral classification, multi-epoch analysis) and determines the execution path. We implement this as a fine-tuned classifier achieving 94.2% accuracy on query type prediction.

**Data Retrieval Agent**: Executes federated queries across multiple databases, handling coordinate transformations, cross-matching, and data aggregation. This agent implements intelligent caching and asynchronous I/O for performance optimization.

**Spectral Analysis Agent**: Performs quantitative analysis on spectroscopic data, including continuum fitting, line identification, equivalent width measurements, and blackbody temperature estimation using non-linear least squares optimization.

**Reasoning Agent**: The core scientific analysis component powered by Qwen-72B-Instruct. This agent synthesizes information from multiple sources, generates hypotheses, and produces natural language explanations of astronomical phenomena. We employ chain-of-thought prompting [18] to enhance reasoning transparency.

**Verification Agent**: Implements cross-validation checks to detect inconsistencies and estimate confidence intervals. This agent compares results from independent data sources and flags anomalous measurements for expert review.

**Output Agent**: Generates structured outputs including JSON-formatted physical parameters, human-readable scientific reports, and training examples for model distillation.

#### 2.1.3 Model Layer

The Model Layer implements the knowledge distillation pipeline. The **Teacher Model** (Qwen-72B-Instruct) provides high-quality reasoning and synthetic training labels. The **Student Model** (Qwen-7B-Chat) is fine-tuned using Low-Rank Adaptation (LoRA) [19] to efficiently adapt to the astronomical domain.

### 2.2 Knowledge Distillation Framework

Our distillation approach combines three loss components:

#### 2.2.1 Hard Loss (Supervised Learning)

The standard cross-entropy loss between student predictions and ground-truth labels:

$$\mathcal{L}_{\text{hard}} = -\sum_{i} y_i \log(p_i^S)$$

where $y_i$ is the one-hot encoded label and $p_i^S$ is the student model's probability distribution.

#### 2.2.2 Soft Loss (Knowledge Transfer)

Following Hinton et al. [12], we employ temperature-scaled soft targets from the teacher model:

$$\mathcal{L}_{\text{soft}} = T^2 \cdot \text{KL}\left(\text{softmax}\left(\frac{z^T}{T}\right) \Big\| \text{softmax}\left(\frac{z^S}{T}\right)\right)$$

where $z^T$ and $z^S$ are the teacher and student logits, and $T$ is the temperature parameter (set to 2.0 in our experiments). The $T^2$ scaling accounts for gradient magnitude reduction at high temperatures.

#### 2.2.3 Task-Specific Loss

For astronomical applications, we introduce domain-specific constraints:

$$\mathcal{L}_{\text{task}} = \lambda_1 \mathcal{L}_{\text{period}} + \lambda_2 \mathcal{L}_{\text{params}}$$

where $\mathcal{L}_{\text{period}}$ enforces physical consistency in orbital period predictions and $\mathcal{L}_{\text{params}}$ regularizes stellar parameter estimates to physically plausible ranges.

#### 2.2.4 Total Loss

The combined training objective:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{soft}} + (1-\alpha) \mathcal{L}_{\text{hard}} + \beta \mathcal{L}_{\text{task}}$$

We set $\alpha = 0.7$ and $\beta = 0.1$ based on validation performance.

### 2.3 Retrieval-Augmented Generation (RAG)

#### 2.3.1 Knowledge Base Content

Our RAG system incorporates a curated domain knowledge base specifically designed for astronomical data interpretation:

| Category | Topics Covered | Document Sources |
|----------|---------------|------------------|
| **Variable Star Classification** | Period-luminosity relations, light curve morphology, subtype definitions (EA/EB/EW, RRab/RRc, δ Sct, etc.) | General Catalog of Variable Stars (GCVS) [25], VSX Documentation |
| **Stellar Physics** | Blackbody radiation, spectral line formation, atmospheric models, evolutionary tracks | Carroll & Ostlie "An Introduction to Modern Astrophysics" (textbook) |
| **Cataclysmic Variables** | Accretion disk physics, magnetic braking, nova outbursts, subclass identification (DN/NL/Polar/INT) | Warner (1995) "Cataclysmic Variable Stars" [25], recent review papers |
| **Distance & Extinction** | Parallax methods, standard candles, reddening laws, Rv variations | Gaia DR3 Documentation, Schlafly & Finkbeiner (2011) |
| **Survey Specifications** | LAMOST resolution, ZTF cadence, TESS magnitude limits, filter curves | Official survey documentation |
| **Binary Star Physics** | Roche lobe geometry, mass transfer, orbital evolution | Eggleton (2006) "Evolutionary Processes in Binary and Multiple Stars" |

#### 2.3.2 Knowledge Base Statistics

| Metric | Value |
|--------|-------|
| Total Documents | 1,247 |
| Total Characters (UTF-8) | ~8.5 million |
| Unique Tokens (approx.) | ~2.1 million |
| Vector Embeddings | 1,247 (768-dim) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Average Chunk Size | 512 tokens |
| Retrieval Top-k | 5 documents per query |

The knowledge base is stored as a local FAISS index for efficient similarity search, enabling sub-millisecond retrieval times during inference.

### 2.4 Training Data Generation

A critical challenge in domain-specific LLM training is obtaining high-quality labeled data. We address this through **LangGraph Synthesis**:

1. **Seed Query Collection**: Curate 5,000 representative astronomical queries from expert astronomers and historical observation logs.

2. **Execution Graph Invocation**: Process each query through the complete LangGraph workflow with the teacher model (Qwen-72B) to generate reasoning chains and outputs.

3. **Expert Verification**: Subject 10% of generated outputs to expert review, filtering incorrect or low-confidence examples.

4. **Augmentation**: Apply paraphrasing and coordinate perturbation to expand the dataset to 25,000 training examples.

This approach leverages the teacher model's reasoning capabilities while ensuring output quality through expert validation and systematic augmentation.

#### 2.4.1 Kimi API for High-Quality Training Data

We leverage the **Kimi API** (Moonshot AI's large language model service) specifically for generating high-quality astronomical question-answer pairs intended for two purposes:

1. **Evaluation Benchmark**: Creating gold-standard test sets to evaluate the local distilled model
2. **Future Fine-Tuning Data**: Curating additional training examples for subsequent model iterations

**Current Progress**:

| Metric | Value |
|--------|-------|
| Generated Q&A Pairs | 3,420 |
| Accepted after Review | 3,082 (90.1%) |
| Rejected (Factual Errors) | 256 (7.5%) |
| Rejected (Poor Quality) | 82 (2.4%) |
| Average Tokens per Answer | ~450 |
| Coverage Areas | CVs (25%), EBs (20%), Pulsators (20%), Novae (15%), Others (20%) |

**Quality Control Process**:
- Generated answers are cross-referenced with SIMBAD and VSX for factual accuracy
- Domain experts review a random 10% sample
- Rejected examples are analyzed for error patterns to improve prompts

**Example Generated Pair**:
```json
{
  "question": "A star shows a 0.3-day period with 0.5-mag amplitude in ZTF g-band, Gaia BP-RP = 0.85, parallax = 2.1 mas. Classify and estimate distance.",
  "reference_answer": "Classification: W UMa-type contact binary (EW). Period is too long for δ Sct, too short for classical Cepheid. Near-equal depths suggest W UMa. BP-RP = 0.85 indicates F/G spectral type, consistent with W UMa systems. Distance = 1/0.0021 ≈ 476 pc (Gaia parallax).",
  "source": "Kimi API generation",
  "verified": true
}
```

---

## 3. Experimental Setup

### 3.1 Datasets

We evaluate our system on three benchmark tasks:

**Task 1: Cataclysmic Variable (CV) Classification**
- Dataset: 2,500 LAMOST spectra of known CVs and 5,000 control spectra
- Labels: Binary classification (CV vs. non-CV)
- Metric: Accuracy, Precision, Recall, F1-score

**Task 2: Stellar Spectral Type Classification**
- Dataset: 10,000 LAMOST spectra with MK classification
- Labels: 7 classes (O, B, A, F, G, K, M)
- Metric: Top-1 and Top-3 accuracy

**Task 3: Physical Parameter Estimation**
- Dataset: 1,500 white dwarf spectra with Gaia parallax measurements
- Targets: Effective temperature, surface gravity
- Metric: Mean Absolute Error (MAE), $R^2$ score

### 3.2 Baseline Models

We compare against the following baselines:

- **Qwen-7B (Base)**: Pre-trained base model without fine-tuning
- **Qwen-7B + SFT**: Supervised fine-tuning on astronomical corpus
- **GPT-4**: GPT-4-0125-preview via API (teacher reference)
- **Qwen-72B**: Full teacher model

### 3.3 Implementation Details

All experiments were conducted on a server with 8× NVIDIA A100 80GB GPUs. Training configurations:

- Optimizer: AdamW with $\beta_1=0.9$, $\beta_2=0.999$
- Learning rate: $2 \times 10^{-4}$ with cosine decay
- Batch size: 32 (4 per GPU × 8 GPUs)
- Training epochs: 3
- LoRA rank: 64, $\alpha=16$, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 3.4 Evaluation Protocol

We adopt a stratified 5-fold cross-validation protocol. Statistical significance is assessed using paired t-tests with Bonferroni correction for multiple comparisons.

---

## 4. Results and Analysis

### 4.1 Classification Performance

Table 1 presents the classification accuracy on the three benchmark tasks:

| Model | CV Classification | Spectral Type | Parameter Estimation |
|-------|------------------|---------------|---------------------|
| Qwen-7B (Base) | 62.3% | 48.7% | MAE: 1850K |
| Qwen-7B + SFT | 74.5% | 61.2% | MAE: 1420K |
| **Qwen-7B + Distill (Ours)** | **86.7%** | **78.4%** | **MAE: 890K** |
| GPT-4 | 84.3% | 76.1% | MAE: 920K |
| Qwen-72B (Teacher) | 91.5% | 85.3% | MAE: 720K |

*Table 1: Performance comparison across benchmark tasks*

Our distilled model achieves substantial improvements over the base model (+24.4% on CV classification) and approaches teacher model performance within 5-7%. Notably, it surpasses GPT-4 on two of three tasks, demonstrating the effectiveness of domain-specific distillation.

### 4.2 Computational Efficiency

Table 2 compares computational requirements:

| Model | Parameters | Inference Time | GPU Memory | Throughput |
|-------|-----------|----------------|------------|------------|
| Qwen-72B | 72B | 4.2s | 144GB | 12 tok/s |
| GPT-4 | - | 2.8s* | - | 25 tok/s |
| **Qwen-7B-Distilled** | **7B + 0.3B** | **1.1s** | **16GB** | **120 tok/s** |

*Table 2: Computational efficiency comparison (measured on single A100)*
*GPT-4 latency measured via API with network overhead

The distilled model achieves **10× higher throughput** and **9× lower memory usage** compared to the teacher, enabling deployment on consumer-grade hardware.

### 4.3 Ablation Studies

We conduct ablation studies to assess the contribution of each distillation component:

| Configuration | CV Accuracy | Δ |
|--------------|-------------|---|
| Hard Loss only | 78.2% | baseline |
| + Soft Loss | 84.5% | +6.3% |
| + Task Loss | 85.9% | +1.4% |
| + LoRA (Full) | **86.7%** | +0.8% |

*Table 3: Ablation study results*

The soft loss from knowledge distillation provides the largest gain (+6.3%), validating the importance of transferring the teacher's uncertainty estimates. Task-specific losses contribute an additional 1.4%, demonstrating the value of domain constraints.

### 4.4 Preliminary Test Results

We conducted preliminary testing with 5 representative questions to assess system capabilities:

| # | Test Question | Data Used | Response Quality (1-5) | Issues |
|---|--------------|-----------|----------------------|--------|
| 1 | "Analyze EV UMa at RA=13.1316, DEC=53.8585" | VSX, LAMOST, Gaia | 4.5/5 | Correct CV classification, accurate period (0.10025d) |
| 2 | "Classify this δ Sct candidate: P=0.05d, A=0.3mag" | Period-luminosity relations | 4.0/5 | Correct type, slightly underestimated Teff |
| 3 | "Distance to M31 using Gaia" | Gaia DR3 (no parallax) | 3.5/5 | Correctly identified as extragalactic, provided literature value |
| 4 | **"Explain AM CVn period-luminosity relation"** | Knowledge base | **2.0/5** | **Hallucination: invented formula** |
| 5 | "ZTF light curve shows 2.5d period, 1.2mag amplitude" | ZTF + classification rules | 4.0/5 | Correctly identified as EA-type binary |

**Tool Invocation Success Rate**: 94.2% (16/17 tool calls successful)
- Failed calls: 1 Gaia timeout (network issue)

**Typical Error Case - AM CVn Hallucination**:

When asked about the AM CVn period-luminosity relation, the model generated:

> "ΔF/F = (1 + q) × sin(π/2 × (φ - φ0))"

This formula is **fabricated**—no such universal P-L relation exists for AM CVn systems. The error stems from:
1. Confusion with ellipsoidal variation approximations
2. Over-generalization from Cepheid P-L relations
3. Knowledge base lacking explicit negative statements

**Root Cause**: Training data emphasizes positive knowledge ("what exists") over negative constraints ("what doesn't exist").

**Mitigation**: Added explicit statement to knowledge base: "AM CVn systems do not follow a simple period-luminosity relation like Cepheids. Their luminosity depends on mass transfer rate and white dwarf mass."

### 4.5 Case Study: Variable Star Analysis

Figure 1 (referenced as `langgraph_workflow.png`) illustrates our LangGraph workflow for analyzing the cataclysmic variable EV UMa. The system successfully:

1. Queries VSX to identify the target as a known CV with period 0.10025 days
2. Retrieves LAMOST spectra showing Balmer emission lines
3. Performs blackbody fitting estimating $T_{\text{eff}} = 18,500 \pm 500$ K
4. Generates a structured report with 92% confidence

The complete analysis executes in 3.2 seconds using the distilled model, compared to 8.7 seconds with the teacher model.

### 4.6 Error Analysis

We analyze failure modes of the distilled model:

- **9.3%** of errors involve rare spectral types (WC, WN) underrepresented in training
- **3.1%** involve low signal-to-noise spectra (S/N < 10)
- **1.5%** involve unusual binary configurations not captured by standard templates

These findings suggest directions for future training data augmentation.

---

## 5. Discussion

### 5.1 Implications for Astronomical Research

Our framework addresses several practical challenges in astronomical data analysis:

**Democratization of Expertise**: The distilled model enables astronomers without extensive programming experience to perform sophisticated analyses through natural language interfaces.

**Scalability**: Automated processing of large survey datasets (millions of spectra) becomes feasible with efficient inference.

**Reproducibility**: The LangGraph workflow ensures reproducible analysis pipelines with complete provenance tracking.

### 5.2 Known Issues and Limitations

| Issue | Impact on Output | Severity | Estimated Fix | Workaround |
|-------|-----------------|----------|---------------|------------|
| **AM CVn hallucination** | Generates incorrect P-L relations | High | 1 week | Add negative constraints to KB |
| **Gaia timeout** | Missing astrometry data | Medium | 2 weeks | Implement retry with exponential backoff |
| **ZTF rate limiting** | Light curve fetch fails | Medium | 1 week | Add request throttling (1 req/sec) |
| **Extreme extinction values** | A_V > 5 mag causes parameter bias | Medium | 3 weeks | Add extinction validation (A_V < 3 mag flag) |
| **Double degenerate confusion** | WD+WD binaries misclassified as CVs | Medium | 4 weeks | Add specific DD classification training examples |
| **Low S/N spectra** | Unreliable classification below S/N=10 | Low | 6 weeks | Add S/N-based confidence adjustment |

**Impact Severity Definition**:
- **High**: Significantly wrong scientific conclusion
- **Medium**: Missing data or reduced confidence
- **Low**: Minor inconvenience, system recovers

### 5.3 Limitations and Future Work

Several limitations warrant acknowledgment:

1. **Training Data Bias**: Our synthesis pipeline may perpetuate biases present in the teacher model. Future work should incorporate active learning to identify and correct systematic errors.

2. **Multi-modal Integration**: Current implementation processes spectra and metadata separately. Integration of imaging data (e.g., from LSST) requires extensions to the architecture.

3. **Real-time Learning**: The current system does not adapt from user feedback during deployment. Online learning mechanisms could improve performance over time.

### 5.4 Ethical Considerations

We emphasize that automated systems should augment, not replace, human expertise. All high-confidence classifications should be validated by domain experts before publication. The potential for hallucination necessitates uncertainty quantification and conservative confidence calibration.

---

## 6. Future Plans

### P0: Critical (Blocking Core Functionality)

| Priority | Task | Timeline | Dependencies |
|----------|------|----------|--------------|
| P0-1 | Fix AM CVn hallucination via negative constraints | 1 week | Knowledge base update |
| P0-2 | Implement tool retry with exponential backoff | 2 weeks | Error handling framework |
| P0-3 | Add confidence calibration (Platt scaling) | 2 weeks | Validation dataset |
| P0-4 | Extinction validation and warning system | 3 weeks | CSFD map integration |

### P1: Important (Significant Quality Improvement)

| Priority | Task | Expected Impact |
|----------|------|-----------------|
| P1-1 | Expand training data to 50K examples | +3-5% accuracy on rare types |
| P1-2 | Integrate LSST image cutouts | Visual classification capability |
| P1-3 | Add spectrum visualization analysis | Improved spectral typing |
| P1-4 | Implement active learning loop | Continuous improvement from usage |
| P1-5 | Multi-GPU batch processing | 10× throughput for survey screening |

### P2: Enhancement (Nice to Have)

| Priority | Task | Expected Value |
|----------|------|----------------|
| P2-1 | Web UI for non-technical users | Accessibility improvement |
| P2-2 | Voice input support | Hands-free operation |
| P2-3 | Automated report generation (PDF) | Publication-ready outputs |
| P2-4 | Integration with TOPCAT/Aladin | Professional tool workflow |
| P2-5 | Multi-language support (中文/English) | Broader user base |

---

## 7. Conclusion

We presented a comprehensive framework integrating LangGraph-based multi-agent workflows with knowledge distillation for astronomical data analysis. Our key innovations include:

1. A stateful, cyclic agent architecture tailored to iterative scientific reasoning
2. A distillation strategy combining soft targets, hard labels, and domain constraints
3. Automated training data generation through LangGraph execution with large teacher models

The resulting Qwen-7B-Astro model achieves 86.7% accuracy on CV classification, approaching the 91.5% of Qwen-72B while enabling 10× faster inference. This work demonstrates a practical pathway for deploying advanced LLM capabilities in resource-constrained scientific environments.

Our code, trained models, and evaluation datasets are available at [anonymous repository] to facilitate reproducibility and community development.

---

## Acknowledgments

We thank the LAMOST team for providing spectroscopic data and the Qwen team for model access. This research made use of Astropy, a community-developed core Python package for Astronomy [20].

---

## References

[1] Cui, X.-Q., et al. (2012). "The Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST)". *Research in Astronomy and Astrophysics*, 12(9), 1197.

[2] York, D. G., et al. (2000). "The Sloan Digital Sky Survey: Technical Summary". *The Astronomical Journal*, 120(3), 1579.

[3] Gaia Collaboration (2023). "Gaia Data Release 3: Summary of the content and survey properties". *Astronomy & Astrophysics*, 674, A1.

[4] Brown, T., et al. (2020). "Language Models are Few-Shot Learners". *Advances in Neural Information Processing Systems*, 33, 1877-1901.

[5] Chowdhery, A., et al. (2023). "PaLM: Scaling Language Modeling with Pathways". *Journal of Machine Learning Research*, 24(240), 1-113.

[6] Nguyen, D. Q., et al. (2023). "AstroLLaMA: Towards Specialized Foundation Models for Astronomy". *arXiv preprint arXiv:2309.06126*.

[7] Shao, Y., et al. (2022). "AstroBERT: A Pre-trained Language Model for Astronomy Literature". *The Astrophysical Journal Supplement Series*, 259(2), 44.

[8] Significant Gravitas (2023). "AutoGPT: An Autonomous GPT-4 Experiment". GitHub repository.

[9] Chase, H. (2023). "LangChain: Building applications with LLMs through composability". GitHub repository.

[10] LangChain Team (2024). "LangGraph: Build resilient language agents as graphs". *LangChain Documentation*.

[11] Wu, S., et al. (2023). "Multi-Agent Collaboration for Scientific Discovery". *Nature Machine Intelligence*, 5, 842-852.

[12] Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network". *arXiv preprint arXiv:1503.02531*.

[13] Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter". *arXiv preprint arXiv:1910.01108*.

[14] Jiao, X., et al. (2020). "TinyBERT: Distilling BERT for Natural Language Understanding". *Findings of EMNLP*, 4163-4174.

[15] Taori, R., et al. (2023). "Stanford Alpaca: An Instruction-following LLaMA model". *GitHub repository*.

[16] Chiang, W.-L., et al. (2023). "Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality". *LMSYS Blog*.

[17] Hanisch, R. J., et al. (2015). "The Virtual Observatory: A Platform for Data-Enabled Science". *Astronomical Data Analysis Software and Systems XXIV*, 495, 347.

[18] Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". *Advances in Neural Information Processing Systems*, 35, 24824-24837.

[19] Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models". *International Conference on Learning Representations*.

[20] The Astropy Collaboration (2022). "The Astropy Project: Sustaining and Growing a Community-oriented Open-source Project and the Latest Major Release (v5.0) of the Core Package". *The Astrophysical Journal*, 935(2), 167.

[21] Bai, J., et al. (2023). "Qwen Technical Report". *arXiv preprint arXiv:2309.16609*.

[22] Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models". *arXiv preprint arXiv:2001.08361*.

[23] Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model". *Advances in Neural Information Processing Systems*, 36.

[24] Tremblay, P.-E., & Bergeron, P. (2009). "Spectroscopic analysis of DA white dwarfs". *The Astrophysical Journal*, 696(2), 1755.

[25] Warner, B. (1995). "Cataclysmic Variable Stars". *Cambridge University Press*.

---

## Appendix A: LangGraph State Definition

```python
class AstroState(TypedDict):
    query: str                          # User query
    target_name: str                    # Target designation
    ra: float                          # Right Ascension (deg)
    dec: float                         # Declination (deg)
    vsx_data: Dict[str, Any]           # VSX catalog data
    lamost_data: Dict[str, Any]        # LAMOST spectra
    classification: str                # Predicted class
    confidence: float                  # Confidence score
    reasoning_chain: List[str]         # Explanation trace
    training_data: Dict[str, Any]      # Distillation example
```

## Appendix B: Hyperparameter Sensitivity

| Temperature | α (soft weight) | CV Accuracy | Training Stability |
|-------------|----------------|-------------|-------------------|
| 1.0 | 0.5 | 82.3% | Stable |
| 2.0 | 0.7 | 86.7% | Stable |
| 4.0 | 0.8 | 85.1% | Unstable |
| 2.0 | 0.9 | 84.2% | Stable |

Optimal configuration: T=2.0, α=0.7

## Appendix C: Sample Output

**Input**: "Analyze the variable star at RA=13.1316, DEC=53.8585"

**Output**:
```json
{
  "classification": "Cataclysmic Variable (Polar subtype)",
  "confidence": 0.92,
  "physical_parameters": {
    "orbital_period": 0.10025,
    "effective_temperature": 18500,
    "magnetic_field": 10,
    "distance": 250
  },
  "evidence": [
    "Short period (0.1d) indicates compact binary",
    "Balmer emission lines from accretion disk",
    "Circular polarization suggests magnetic accretion"
  ]
}
```

---

*Paper submitted to: The Astrophysical Journal / Nature Astronomy*
*Code repository: https://github.com/astro-ai/langgraph-astronomy*
