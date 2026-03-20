# maoAstro LLM

Astronomy domain-specific LLM trained on 20,000+ QA pairs from astrophysics papers.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the model with RAG
python start_maoastro_with_simple_rag.py
```

## Project Structure

```
maoAstro_llm/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
│
├── Core Scripts (15 files)
│   ├── generate_astronomy_qa_hybrid.py    # ⭐ Generate QA pairs from PDFs
│   ├── train_alternative_model.py         # ⭐ Train LLM (Qwen/Llama)
│   ├── start_maoastro_with_simple_rag.py  # ⭐ Start model with RAG inference
│   ├── export_astrosage_simple.py         # Export to Ollama format
│   │
│   ├── evaluate_model.py                  # Model evaluation
│   ├── test_manual_eval.py                # Manual testing
│   ├── test_api.py                        # Test API availability
│   ├── analyze_qa_results.py              # Analyze QA dataset
│   │
│   ├── use_astrosage_with_rag.py          # Use Ollama + RAG
│   ├── check_rag_knowledge.py             # Check RAG knowledge base
│   ├── astro_qa_dataset.py                # Rule-based QA generator
│   ├── generate_report.py                 # Generate project report
│   ├── clean_for_github.py                # Clean project for GitHub
│   └── download_data.py                   # Download astronomical data
│
├── src/                               # 🌟 33 Astronomy Analysis Tools
│   ├── unified_astro_query.py         # ZTF/TESS/LAMOST/SDSS/Gaia unified query
│   ├── lightcurve_processor.py        # Light curve analysis
│   ├── spectrum_analyzer.py           # Spectral analysis
│   ├── hr_diagram_plotter.py          # HR diagram plotting
│   ├── sed_plotter.py                 # SED plotting
│   ├── astro_analyzer.py/v2           # Comprehensive celestial analysis
│   ├── complete_astro_download.py     # Multi-catalog data download
│   └── ... (25 more tools)
│
├── train_qwen/                        # Training system
│   ├── train_with_qwen25.py          # Qwen2.5-7B LoRA training
│   ├── inference.py                  # Model inference
│   ├── merge_lora.py                 # Merge LoRA weights
│   ├── data/qwen_train.json          # Training data (20,609 QA pairs)
│   └── astrosage_export/             # Export scripts
│
├── rag_system/                        # RAG retrieval system
│   ├── vector_store/chroma_store.py  # Vector database
│   ├── inverted_index/keyword_index.py  # Keyword search
│   └── retrieval/hybrid_retriever.py # Hybrid retrieval
│
├── data/                              # Data directory
│   ├── papers/                       # PDF papers
│   ├── qa_hybrid/                    # Generated QA pairs
│   └── processed/                    # Processed data
│
└── output/                            # Output directory
    └── qa_dataset.json               # Final QA dataset
```

## Models

### Option 1: Train Your Own (Qwen2.5-7B)
- Base: Qwen2.5-7B-Instruct
- Training: LoRA (rank=64, alpha=16)
- Data: 20,609 QA pairs from 242 papers

### Option 2: AstroSage-LLaMA-3.1-8B (Recommended)
From [astromlab.org](https://astromlab.org/ollama.html) - Pre-trained astronomy LLM

```bash
# Download from HuggingFace
# https://huggingface.co/astromlab/AstroSage-LLaMA-3.1-8B

# Or use with Ollama
ollama pull astromlab/astrosage-llama3.1-8b
```

## Tools & Utilities

### Data Generation
```bash
# Generate QA pairs from PDFs (rule-based + API)
python generate_astronomy_qa_hybrid.py --input ./papers --output ./output

# Rule-based QA generator (no API needed)
python astro_qa_dataset.py

# Check RAG knowledge base
python check_rag_knowledge.py
```

### Model Training
```bash
# See train_qwen/README.md for training guide
cd train_qwen && python train_with_qwen25.py

# Alternative models
python train_alternative_model.py --model qwen
```

### Inference & Usage
```bash
# Start with RAG
python start_maoastro_with_simple_rag.py

# Use with Ollama + RAG (AstroSage model)
python use_astrosage_with_rag.py
```

### Astronomy Data Analysis (src/)
```bash
# Unified query for ZTF, TESS, LAMOST, SDSS, Gaia
python src/unified_astro_query.py

# Light curve processing
python src/lightcurve_processor.py

# Spectrum analysis
python src/spectrum_analyzer.py

# HR diagram plotting
python src/hr_diagram_plotter.py

# Complete analysis pipeline
python src/complete_analysis.py
```

### Evaluation & Analysis
```bash
# Model evaluation
python evaluate_model.py --model_path ./train_qwen/output_qwen25/merged_model

# Manual test with reference answers
python test_manual_eval.py

# Analyze QA results
python analyze_qa_results.py --input ./output/qa_hybrid

# Generate project report
python generate_report.py
```

### Utilities
```bash
# Clean project for GitHub
python clean_for_github.py

# Export to Ollama format
python export_astrosage_simple.py

# Download astronomical data
python download_data.py
```

## API Configuration

```bash
export MOONSHOT_API_KEY="YOUR_API_KEY_HERE"
```

## Hardware Requirements

- GPU: NVIDIA RTX 3080 Ti or better (12GB+ VRAM)
- RAM: 32GB+
- Storage: 50GB+ for models and data

## License

MIT License

---

**Related Projects:**
- [AstroMLab](https://astromlab.org/) - Astronomy AI Research Lab
- [AstroSage Model](https://astromlab.org/ollama.html) - Pre-trained astronomy LLM
