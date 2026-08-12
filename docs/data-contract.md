# Data contract

## Episode input

每个推理 episode 是一个目录，其中 `metadata.json` 至少包含：

```json
{
  "episode_id": "clip_001__win0000",
  "source": "ego4d_v2_long_window",
  "task_hint": null,
  "cameras": [
    {"name": "main", "role": "main", "path": "window.mp4", "time_offset_sec": 0.0}
  ]
}
```

相机路径可相对 episode 目录，也可为绝对路径。多相机输入需要共享时间基准，并用
`time_offset_sec` 表达偏移。

## Window annotation

规范化后的 `annotations.jsonl` 每行包含视频级描述、一个或多个 subtask、模型与 prompt
版本、采样 FPS、重试次数和质量 flags。每个 subtask 的核心字段为：

```json
{
  "start_time_sec": 3.2,
  "end_time_sec": 8.9,
  "action": "open",
  "fine_action": "open",
  "object": "drawer",
  "instruction": "Open the drawer.",
  "training_eligible": true,
  "quality_flags": []
}
```

`action` 使用闭集粗粒度词表；`fine_action`、`object` 和 `instruction` 允许开放词表。

## Final segment

`final_timeline.jsonl` 的时间是相对原始长视频的全局时间：

```json
{
  "segment_id": "ego4d_v2__clip_001__seg0007",
  "clip_uid": "clip_001",
  "start_sec": 83.2,
  "end_sec": 89.7,
  "duration_sec": 6.5,
  "action": "open",
  "fine_action": "open",
  "object": "drawer",
  "label": "Open the drawer.",
  "training_eligible": true,
  "quality_flags": [],
  "label_source": "qwen_visual_only",
  "narration_used": false,
  "segment_clip_path": ".../seg0007.mp4",
  "export_ok": true
}
```

不变量：同一 `clip_uid` 内按时间排序、无重叠、无空洞、首段从 0 开始、末段到 source
duration、每段 `duration_sec >= 3.0`。只有 `training_eligible=true` 且 `export_ok=true` 的行
进入 `training_segments.jsonl`。

## Long-video summary

`video_summaries.jsonl` 每条 source video 一行，包含：

- `long_video_caption` / `summary_caption`
- `summary_caption_source`
- `summary_caption_generation`
- `representative_captions`
- `segment_count`, `training_segment_count`, `review_segment_count`
- `label_source=qwen_visual_only`
- `narration_used=false`

当前 summary 由按时间排序的 Qwen window captions 确定性组合；没有 window caption 时回退到
segment labels，并保留 source 字段以便后续替换成独立 summary 模型。
