#!/usr/bin/env bash
# End-to-end: PDF -> dataset -> baseline -> RAG -> LoRA SFT -> evaluation table.
#   bash scripts/run_pipeline.sh data/sample_troubleshooting_guide.pdf
#   METHOD=qlora BASE_MODEL=Qwen/Qwen3-0.6B bash scripts/run_pipeline.sh data/my_company.pdf
set -euo pipefail
PDF=${1:-data/sample_troubleshooting_guide.pdf}
METHOD=${METHOD:-lora}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
cd "$(dirname "$0")/.."

[ -f "$PDF" ] || python scripts/make_sample_pdf.py
python src/pdf_extract.py --pdf "$PDF"
python src/build_dataset.py --method "${DATASET_METHOD:-heuristic}"
python src/rag.py build

python src/eval_model.py --model "$BASE_MODEL" --name 1-base
python src/eval_model.py --model "$BASE_MODEL" --rag-index outputs/rag_index --name 2-base+rag

python src/train_sft.py --method "$METHOD"
python src/eval_model.py --model "outputs/sft-$METHOD" --name "3-sft-$METHOD"
python src/eval_model.py --model "outputs/sft-$METHOD" --rag-index outputs/rag_index --name "4-sft-$METHOD+rag"

python src/eval_model.py --compare "results/*_summary.json"
