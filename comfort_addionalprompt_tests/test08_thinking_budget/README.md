# test07 思考预算对照实验

容器入口沿用 test07 的 Docker 镜像、HF 缓存、Python 依赖缓存、GPU 映射和几何数据挂载。
直接重放 test07 9B thinking CSV 中的 144 条原始 prompt（包括原始选项顺序），在原始输出格式指令与选项之前插入长度指令。
默认加载 Qwen/Qwen3.5-9B 一次，依次运行 baseline、shortest、approx、cap、explicit_end。
baseline 不追加任何指令。所有组使用相同采样参数、样本种子和生成上限。

在项目根目录运行（宿主机 GPU 0，容器内映射为 GPU 0）：

```bash
COMFORT_ADD_PROMPT_GPU=0 bash comfort_addionalprompt_tests/test08_thinking_budget/run_docker.sh
```

先测试固定随机抽样的 16 张图片：

```bash
COMFORT_ADD_PROMPT_GPU=0 bash comfort_addionalprompt_tests/test08_thinking_budget/run_docker.sh \
  --limit 16 --output-dir comfort_addionalprompt_tests/test08_thinking_budget/results_smoke
```

仅测试指定 prompt，或更换模型，必须使用独立输出目录：

```bash
COMFORT_ADD_PROMPT_GPU=1 bash comfort_addionalprompt_tests/test08_thinking_budget/run_docker.sh \
  --model-id Qwen/Qwen3.5-4B --variants approx cap explicit_end \
  --output-dir comfort_addionalprompt_tests/test08_thinking_budget/results_4b
```

默认 `--budget 1024 --max-new-tokens 4096`。1024 是提示词软预算，4096 是整个生成的硬上限，
并非 test07 历史运行的长上限；此处 baseline 与新增 prompt 使用同一上限作公平对照。
如需与 test07 的 20480 配置对齐，传入 `--max-new-tokens 20480`，并使用新输出目录。
脚本沿用仓库 thinking backend 的 temperature=0.6、top_p=0.95、top_k=20。
模型默认明确为 9B；test07 现有 shell 的 thinking 分支写的是 4B，按实际对照模型选择。

结果位于 `results/`：
- 各组对应的 `<variant>.jsonl`：逐样本 prompt、输出、标签、实际生成 token ID 和长度指标。
- `summary.json`：准确率、完整回答率、预算内比例、触顶率、平均/P50/P90 思考长度，以及正确且完整且预算内的比例。
- `config.json`、`prompt_preview.json`：配置与完整提示词预览。

相同命令自动跳过已写入 JSONL 的样本；改变参数必须换目录。不要同时运行写入同一目录的进程。
思考长度按实际生成 ID 到第一个 `</think>` 之前计数，不含输入模板的 `<think>`；
没有闭合标记的输出按未完成记录，其长度不代表完整思考所需长度。

OPSD 当前教师对学生轨迹打分，不单独生成教师 CoT。这里检验的是带几何信息的 test07 上的
prompt 效果，不能直接保证只看图的 student rollout 也能控制在 1024 token；后续需独立验证。
若 OPSD 的 max-completion-length=1024，它限制整个输出，思考预算需要给最终答案和结束符留余量。
不要用最终 held-out test split 选择 prompt；应在验证集选择后固定配置。

不加载模型的本地检查：

```bash
bash comfort_addionalprompt_tests/test08_thinking_budget/run_qwen35.sh \
  --dry-run --limit 16 --output-dir /tmp/comfort-budget-preview
```

## 输出格式指令置后的版本（v2）

排列顺序：原始几何信息 → 原始问题 → 长度约束 → 原始 ONLY the letter 指令 → 原始 A–D 选项。
baseline 原样重放，不添加或移动内容。approx、cap 保留原来的措辞，便于比较位置变化；
删除 scoped、ceiling；explicit_end 简化为 token 上限与结束 thinking 的要求。
默认共五组，仍不增加解题指导。位置调整是否改善格式遵从，需要实测。

旧进程已经加载旧代码，修改文件不会改变其正在运行的提示词。新实验使用独立目录；
配置记录 prompt_layout，防止不同排列方式续写到同一结果目录。

```bash
COMFORT_ADD_PROMPT_GPU=0 bash comfort_addionalprompt_tests/test08_thinking_budget/run_docker.sh \
  --output-dir comfort_addionalprompt_tests/test08_thinking_budget/results_format_last
```

新增 `shortest`：`Keep your thinking as short as possible while maintaining a correct answer.`
此组不指定 token 数，测试纯简洁性约束；统计仍使用 --budget（默认1024）作为统一评估阈值。
旧版 approx 的53条结果已改名为 `results_length_only/approx_length_after_options_v1.jsonl`，
其原始 config.json 与 prompt_preview.json 保留。新版继续使用独立的 results_format_last 目录。
