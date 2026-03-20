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
├── Core Scripts (10 files)
│   ├── generate_astronomy_qa_hybrid.py    # Generate QA pairs from PDFs (rule + API)
│   ├── train_alternative_model.py         # Train LLM (Qwen/Llama)
│   ├── start_maoastro_with_simple_rag.py  # Start model with RAG inference
│   ├── export_astrosage_simple.py         # Export to Ollama format
│   ├── evaluate_model.py                  # Model evaluation
│   ├── analyze_qa_results.py              # Analyze QA dataset
│   ├── test_manual_eval.py                # Manual testing
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

## Workflow

1. **Generate QA Data**: `python generate_astronomy_qa_hybrid.py --input ./papers --output ./output`
2. **Train Model**: See `train_qwen/README.md`
3. **Inference**: `python start_maoastro_with_simple_rag.py`

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
