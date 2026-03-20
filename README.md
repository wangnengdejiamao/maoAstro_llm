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

# Use with Ollama + RAG
python use_astrosage_with_rag.py
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

# Test API availability
python test_api.py
```

### Utilities
```bash
# Clean project for GitHub (remove large files)
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

## Model Info

- Base: Qwen2.5-7B-Instruct
- Training: LoRA (rank=64, alpha=16)
- Data: 20,609 QA pairs from 242 papers
- Loss: 0.34 → 0.21 (3 epochs)

## License

MIT License
