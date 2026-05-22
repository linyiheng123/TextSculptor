# TextSculptor: Training and Benchmarking Scene Text Editing

[![arXiv](https://img.shields.io/badge/arXiv-2605.21090-b31b1b.svg)](https://arxiv.org/abs/2605.21090)
[![Dataset](https://img.shields.io/badge/Dataset-TextSculpt--Data-yellow.svg)](https://huggingface.co/datasets/dafbgd/TextSculpt-Data)
[![Benchmark](https://img.shields.io/badge/Benchmark-TextSculpt--Bench-blue.svg)](https://huggingface.co/datasets/dafbgd/TextSculpt-Bench)



## Installation

The evaluation scripts require Python 3.10+ and `uv`. For CUDA 12.6:

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install -r requirements.txt
```

## TextSculpt-Bench Format

To evaluate a model, generate edited images with your own pipeline and save them with the required filenames and directory layout below.

Download TextSculpt-Bench from Hugging Face and place it under `benchmark/`:

```bash
huggingface-cli download dafbgd/TextSculpt-Bench \
  --repo-type dataset \
  --local-dir benchmark
```

### Benchmark Inputs

Instruction files are JSONL files under `benchmark/instructions/`:

```text
benchmark/instructions/
  add.jsonl
  remove.jsonl
  replace.jsonl
  hybrid.jsonl
```

Each line is one edit task:

```json
{
  "image_filename": "img_00008.png",
  "image_path": "benchmark/nano_poster/img_00008.png",
  "instruction": "Add four new text elements ...",
  "type": "add",
  "replace_targets": [],
  "add_targets": [{"added_text": "NEW"}],
  "remove_targets": [],
  "expected_edit_word_count": 7
}
```

Required fields:

- `image_filename`: filename that the generated result must use.
- `image_path`: path to the source image, relative to the repository root unless absolute.
- `instruction`: natural-language edit instruction.
- `type`: edit type, one of `add`, `remove`, `replace`, `hybrid`.
- `expected_edit_word_count`: expected number of edited words for TA scoring.

### Expected Generated Image Structure

For a model named `<model_name>`, save generated images as:

```text
benchmark/generated/<model_name>/
  add/
    <image_filename>
  remove/
    <image_filename>
  replace/
    <image_filename>
  hybrid/
    <image_filename>
```

Every generated file must have the same `image_filename` as its corresponding JSONL
record. For example, the output for the sample above should be:

```text
benchmark/generated/<model_name>/add/img_00008.png
```


## Evaluation

Set the evaluation model configuration through environment variables before running
TA/VQ:

```bash
export AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint
export AZURE_OPENAI_API_VERSION=api_version
export AZURE_OPENAI_MODEL=gpt-5.2-2025-12-11
export AZURE_OPENAI_API_KEYS=key
```

Run TA/VQ and background-preservation evaluation for each edit type:

```bash
model=your_model_name
edit_types=(add remove replace hybrid)

for edit_type in "${edit_types[@]}"; do
  mkdir -p benchmark/evaluations/${model}/${edit_type}

  python cal_ta_vq.py \
    --input benchmark/instructions/${edit_type}.jsonl \
    --edited-dir benchmark/generated/${model}/${edit_type} \
    --output benchmark/evaluations/${model}/${edit_type}/ta_vq.json \
    --details benchmark/evaluations/${model}/${edit_type}/ta_vq_details.jsonl

  python cal_bp.py \
    --input benchmark/instructions/${edit_type}.jsonl \
    --edited-dir benchmark/generated/${model}/${edit_type} \
    --output benchmark/evaluations/${model}/${edit_type}/bp.json \
    --details benchmark/evaluations/${model}/${edit_type}/bp_details.jsonl
done
```

Expected evaluation output structure:

```text
benchmark/evaluations/<model_name>/
  add/
    ta_vq.json
    ta_vq_details.jsonl
    bp.json
    bp_details.jsonl
  remove/
    ...
  replace/
    ...
  hybrid/
    ...
```

The TA/VQ summary file contains aggregate text accuracy and visual quality scores. The
BP summary file contains aggregate MSE, PSNR, and SSIM values; SSIM is used as
the BP score by the overall scorer.

After all per-type evaluations finish, summarize model scores:

```bash
python get_overall_score.py \
  --eval-root benchmark/evaluations \
  --instruction-dir benchmark/instructions \
  --output-txt results/model_summary.txt
```

The overall scorer expects complete `ta_vq_details.jsonl` and
`bp_details.jsonl` files for all four edit types. The final score is the
average of text accuracy, visual quality, and BP.

## Citation

If you find this project useful, please cite:

```bibtex
@article{lin2026textsculptor,
  title={TextSculptor: Training and Benchmarking Scene Text Editing},
  author={Lin, Yiheng and Jiao, Siyu and Lan, Xiaohan and Zhou, Wei and She, Qi and Yu, Fei and Chen, Heyun and Wang, Zhengwei and Chen, Jinghuan and Li, Moran and Yu, Yingchen and Feng, Zijian and Zhao, Yao and Wei, Yunchao and Zhong, Yujie},
  journal={arXiv preprint arXiv:2605.21090},
  year={2026}
}
```
