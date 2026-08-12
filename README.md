# Ego Video Subtask Pipeline

把 5–8 分钟第一视角视频变成可审查的短 subtask 视频、时间边界和视觉 label。长视频先被
覆盖为重叠的推理窗口，VLM 在窗口内给出候选动作，随后在原始时间轴上去重、消解冲突、
补齐未覆盖区域，并重新从原视频导出最终片段。

核心约束：

- 最终视频硬下限为 **3 秒**；短片段会与相邻段合并。
- 不同语义为满足时长而合并时会标记 `merged_for_min_duration`，只能人工复核，不能直接训练。
- 每条长视频生成一个 `long_video_caption` 和一条完整、连续、无重叠时间线。
- Ego4D narration 默认不参与选样、prompt、边界或 label；输出明确记录 `narration_used=false`。
- `idle`、含糊 `other`、未覆盖区间和任何有阻断 flag 的片段进入 review queue。

## Workflow

```text
long ego video
  → overlapping analysis windows
  → visual-only VLM subtask candidates
  → map to source time + reconcile overlaps
  → enforce complete timeline and ≥3 s segments
  → export MP4 + JSONL + long-video caption
  → human review → training manifest
```

分析窗口不是训练片段。默认每个窗口 30 秒、重叠 5 秒；真正的短视频只在全局边界确定后
从原始视频导出。

## Installation

需要 Python 3.10+，以及系统命令 `ffmpeg`、`ffprobe`。

```bash
git clone https://github.com/SII-ZijunX/ego-video-subtask-pipeline.git
cd ego-video-subtask-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

本地 Qwen 推理额外安装：

```bash
pip install -e ".[qwen]"
```

## CPU-only smoke test

Mock backend 不需要 GPU 或模型权重，会自动生成一条 6 秒合成视频并走完 episode pipeline：

```bash
bash examples/run_mock.sh
pytest
```

## Long-form Ego4D run

### 1. 准备重叠分析窗口（CPU）

`--ego4d-root` 应包含 `ego4d.json` 和 `v2/clips/*.mp4`。也可以设置 `EGO4D_ROOT`。

```bash
ego-video-long prepare \
  --ego4d-root /path/to/ego4d \
  --output-dir runs/ego4d_batch_001 \
  --num-videos 10 \
  --min-duration-sec 300 \
  --max-duration-sec 490 \
  --window-sec 30 \
  --overlap-sec 5 \
  --workers 8
```

### 2. 对窗口进行 VLM 标注（GPU 或 API）

先复制示例配置，写入本机模型路径；`.local.yaml` 已被 Git 忽略。

```bash
cp configs/qwen_local.example.yaml configs/qwen_local.local.yaml
ego-video-pipeline annotate-batch \
  --dataset runs/ego4d_batch_001/dataset \
  --output runs/ego4d_batch_001/qwen \
  --config configs/qwen_local.local.yaml \
  --resume
```

如使用 OpenAI-compatible vision endpoint，复制
`configs/openai_compatible.example.yaml`，并通过环境变量提供 key：

```bash
export VLM_API_KEY='...'
```

检查进度：

```bash
ego-video-long status --work-dir runs/ego4d_batch_001
```

### 3. 拼接完整时间线并导出最终短片（CPU）

```bash
ego-video-long finalize \
  --work-dir runs/ego4d_batch_001 \
  --min-final-segment-sec 3 \
  --workers 8
```

不要在生产中使用 `--allow-incomplete`。该选项仅用于调试，窗口未完成时会产生更大的
`uncovered_by_qwen` review 区域。

### 4. 人工审查

```bash
python -m http.server 8000 --directory runs/ego4d_batch_001
```

打开 `http://localhost:8000/review_index.html`。Notebook 环境需要使用平台提供的 8000 端口
代理 URL；`review_index.html` 必须位于 `--directory` 指定目录内，否则会返回 404。

审查内容：边界是否包含完整动作、label 是否只描述可见事实、合并段是否包含多个不同动作、
`idle/other/uncovered` 是否应该重切，以及长视频 caption 是否符合动作顺序。详见
[质量门禁](docs/quality-gates.md)。

## Generic MP4 datasets

ActionNet、RoboMIND、HoloAssist、Sekai 等直接提供 MP4 的数据集可以复用同一条长视频管线。
先准备 JSONL manifest，每行至少包含 `source_clip_path`，并建议提供 `dataset`、`clip_uid`、
`video_uid`、`scenario` 或 `task_hint`：

```json
{"dataset":"actionnet","clip_uid":"01JJ...","source_clip_path":"/data/01JJ.../top/rgb.mp4","task_hint":"optional reference only"}
```

LeRobot v3 等数据集可能把许多 episodes 拼进同一个长 MP4。此时不要把整个容器当作一条
episode；在 manifest 中提供容器内的秒级范围。管线会在分析窗口和最终片段导出时自动加上
该偏移，不需要复制大文件：

```json
{"dataset":"airoa-moma","clip_uid":"episode-000007","source_clip_path":"/data/videos/head/file-000.mp4","source_start_sec":101.25,"source_end_sec":114.75,"reference_caption":"旁路对照文本","task_hint":null}
```

`reference_caption` 可保留用于事后评估；只要 `task_hint` 为 `null`，它不会进入视觉标注
prompt。`source_start_sec/source_end_sec` 必须落在 MP4 容器时长内，episode 时长仍必须满足
`--min-duration-sec/--max-duration-sec`。

然后运行：

```bash
ego-video-long prepare \
  --source-manifest runs/mixed/sources.jsonl \
  --output-dir runs/mixed \
  --min-duration-sec 3 \
  --max-duration-sec 3600 \
  --window-sec 30 \
  --overlap-sec 5
```

后续 `annotate-batch`、`finalize`、≥3 秒硬门禁、long-video caption 和审查页面与 Ego4D
完全一致。MCAP、仅图片、archive-only 或远程 manifest 数据集仍需先转换/解析为本地 MP4，不能仅凭
目录存在假定兼容。

## Main outputs

| File | Meaning |
| --- | --- |
| `source_videos.jsonl` | 输入长视频清单和来源元数据 |
| `analysis_windows.jsonl` | 推理窗口、全局时间和导出状态 |
| `qwen/annotations.jsonl` | 每个窗口的 VLM 原始结构化结果 |
| `candidate_subtasks.jsonl` | 映射回 source time 的全部候选 |
| `final_timeline.jsonl` | 100% 时间轴上的最终短片及 label |
| `final_segments/<clip_uid>/*.mp4` | 最终切分视频 |
| `training_segments.jsonl` | 自动门控后可进入训练的保守集合 |
| `review_queue.jsonl` | 必须人工复核的片段 |
| `video_summaries.jsonl` | 每条长视频的汇总 caption |
| `timeline_summary.json` | 覆盖率、静态率、动作和 flag 统计 |
| `review_index.html` | 可播放的视频审查入口 |

完整字段见 [数据契约](docs/data-contract.md)，流程细节见
[长视频流程](docs/long-video-workflow.md)。

## Generic episode commands

除长视频编排外，底层 CLI 还提供：

```text
annotate, annotate-batch, validate, visualize, stats,
prepare-ego4d, evaluate-references, finalize
```

运行 `ego-video-pipeline --help` 或对应子命令的 `--help` 查看参数。所有 annotate/validate
命令都要求显式传入 `--config`，避免静默使用某台机器的模型路径。

## Reproducibility and release checks

每个 episode 保存配置快照、采样帧和时间戳、prompt、原始响应、解析结果、验证报告、模型元
数据和 Git commit。批处理逐条落盘，支持 `--resume`。

提交前运行：

```bash
python -m compileall video_annotation_pipeline
pytest
rg -n '/inspire/|api[_-]?key\s*:|xuzijun|253108540220' . \
  -g '!README.md' -g '!SECURITY.md'
rg --files | rg '\.(mp4|pt|pth|safetensors|log)$' || true
```

不要提交数据集、生成视频、模型权重、API key、内部绝对路径或审查结果。Ego4D 和模型各自的
许可证不会被本仓库的 MIT 许可证覆盖。

准备发布时再按 [GitHub release checklist](docs/github-release-checklist.md) 补充组织 URL、
maintainer 联系方式和仓库保护规则；当前目录没有连接或推送到任何 GitHub remote。

## License and provenance

MIT licensed. This standalone package was derived from preprocessing work in
the MIT-licensed Microsoft VITRA repository; see [NOTICE.md](NOTICE.md).
