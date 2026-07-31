# claude.md

## 协作规则
1. 需求模糊时，先提问澄清再写代码
2. 尽量不写兼容性代码，除非我主动要求
3. 写代码前先描述方案，等我批准后动手
4. 修改超过3个文件时，先拆成小任务

---

## 项目背景：Spatial Evaluation Research

**研究目标**：评估视觉语言模型（VLM）能否理解以人为中心的空间关系（egocentric spatial relations）。给定图像中人物和物体，VLM能否以图中人的第一视角判断物体在其左/右/前/后？

---

## 数据集结构

### WhatsUp_dataset（源数据，禁止修改）
- 原始数据集，含物体相对于家具的位置图像（无人物），来自WhatsUp benchmark

### Version0_dataset（基准测试集，禁止删除/修改）
- 从 WhatsUp_dataset 衍生，用图像编辑模型向原图插入一个坐着/站着的人物
- 人物朝向：FACE-CAMERA / FACE-LEFT / FACE-RIGHT（三种），每种产生不同的真实空间标签
- **文件名约定**：`{second_object}_{direction}_of_{furniture}_FACE-{view}.png`
- **标签转换**（`spatial_eval/utils.py`）：
  - FACE-CAMERA + right → 正确答案 = left（镜像翻转）
  - FACE-LEFT + right → 正确答案 = behind
  - FACE-RIGHT + right → 正确答案 = front
- **子目录**（不同生成模型）：
  - `Qwen_Image_Edit_2509_v0` — 主要数据集（Qwen-Image-Edit-2509 + Lightning LoRA）
  - `Qwen_Image_Edit_2509_v0_armchair` — 扶手椅变体
  - `Qwen_Image_Edit_2509_v0_chair_2nd` — 无效图片重新生成
  - `Qwen_Image_Edit_2509_v0_table` — 桌子变体（人物站立）
  - `FLUX2_v0` — FLUX2-Klein-9B 生成
  - `kontext_v0` — Kontext（FLUX2变体）生成
- 每个子目录有 `meta_v0.csv`（src_path, dst_path, prompt, expected_direction, second_object, view_type）
- **验证集**：`whatsup_eval/whatsup_image_validation/valide_image_paths.json`（563张通过Qwen3-VL两轮检验的有效图片）

### COMFORT/data/comfort_human_car（合成3D数据集）
- 用 COMFORT 框架（Blender渲染）生成，人物模型 Sophia + 9种物体 × 4方向 × 4摄像角度 = 144张
- 目录结构：`{relation}/{object}__{relation}__{cam_view}/0.png + config.json`
- 生成脚本：`COMFORT/data_generation/generate_dataset.py`，配置：`comfort_human_car_config.py`
- **路径索引**：`comfort_image_paths.json`

---

## 核心评估代码：spatial_eval/（Python package）

### 入口
```
python -m spatial_eval.cli --backend <model> --model_id <hf_id> \
  --image_json whatsup_eval/whatsup_image_validation/valide_image_paths.json --out_csv eval_output/xxx.csv \
  --ask_mode mcq --answer_length short/middle/long
```

### 模块结构
- `cli.py` — 参数解析，组装 backend + asker + eval_runner
- `eval_runner.py` — 遍历图片列表，支持Version0路径解析和COMFORT路径解析两种模式
- `utils.py` — 两个解析器：`parse_relation_from_basename()`（Version0）、`parse_relation_for_COMFORT()`（COMFORT）
- `backends/` — 各VLM适配器，统一接口 `ask(image_path, prompt, max_new_tokens) -> str`
- `prompts/MCQ.py` — 四选一MCQ（随机打乱选项），提取预测字母
- `prompts/YesNo.py` — 5个是/否问题

### 支持的模型（backend key → 模型）
| key | 模型 |
|-----|------|
| `qwen` | Qwen2.5-VL-7B-Instruct |
| `qwen3vl` | Qwen3-VL-8B-Instruct |
| `qwen3-vl-thinking` | Qwen3-VL-8B-Thinking |
| `qwen3.5vl` | Qwen3.5-9B |
| `qwen3vl-logits` | Qwen3-VL-8B（logit打分） |
| `internvl` | InternVL3.5-8B-HF |
| `gemma3` | gemma-3-12b-it |

### MCQ 问题长度变体
- `short`：A. left / B. right / C. front / D. behind
- `middle`：A. on the left / B. on the right / ...
- `long`：完整句子描述

### 输出CSV列
`image_path, second_object, mcq_prompt, correct_relation, correct_letter, opposite_relation, opposite_letter, model_answer, pred_letter`
（同时记录正确答案和对立答案，用于检测模型是否系统性地反向预测）

---

## 评估结果目录

### eval_output/（Version0_dataset，563张图片）
- 文件命名：`mcq_{length}_{model}.csv`
- 分析：`eval_analyse.ipynb`（准确率、混淆矩阵、方向偏差分析）

### comfort_eval_output/（COMFORT数据集，144张图片）
- 同样命名规范
- 分析：`eval_analyse.ipynb`（COMFORT专用路径解析）

---

## 数据集生成笔记本

| 文件 | 用途 |
|------|------|
| `ComfyUI_HF_QwenImageEdit.ipynb` | 用Qwen-Image-Edit-2509生成椅子/桌子图片（主要） |
| `ComfyUI_HF_QwenImageEdit_create_second_view.ipynb` | 扶手椅/桌子变体 |
| `ComfyUI_HF_QwenImageEdit_regenerate_nonvalid.ipynb` | 重新生成无效图片（含VLM验证循环） |
| `whatsup_eval/whatsup_image_validation/validation_by_mLLM.ipynb` | 用Qwen3-VL进行两轮图片质量验证（历史notebook） |
| `whatsup_eval/whatsup_image_validation/get_valid_image_paths.py` | 可复现的Qwen3-VL验证与有效清单生成脚本 |
| `Version0_Dataset_FLUX2.ipynb` | FLUX2-Klein-9B生成 → FLUX2_v0 |
| `Version0_Dataset_QwenImageEdit.ipynb` | Kontext变体生成 → kontext_v0 |
| `FLUX2_create_second_view.ipynb` | FLUX2-Klein-4B生成（实验用） |

---

## 重要文件位置
- `spatial_eval/` — 主评估包
- `COMFORT/data_generation/` — COMFORT数据生成（含 `generate_dataset.py`）
- `COMFORT_human_car_scene_setup/data_generation/` — 开发中的human_car生成代码副本
- `README.md` — 常用命令速查（conda环境激活、数据生成命令）
- `spatial_eval.py` — 旧版独立脚本（已被模块化版本取代）
