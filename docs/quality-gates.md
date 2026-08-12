# Quality gates

## Segment review

逐段检查：

1. 边界包含完整可见动作，且没有吞入明显的下一个动作。
2. label 只描述画面证据，不推断不可见目的、身份、物体状态或未来动作。
3. object 与动作对象一致；无法确认时使用 review，而不是猜测。
4. `compound` 合并段是否应重新切边界或拆成两个可用片段。
5. `idle`、`other`、`uncovered_by_qwen` 是否确实不可训练。
6. 视频可解码，实际时长与 JSON 一致，且不少于 3 秒。

## Long-video review

检查 `long_video_caption` 是否覆盖主要阶段、符合时间顺序、没有 narration 泄漏或事实幻觉。
它是 episode 级索引和检索 caption，不替代 segment label。

## Automatic release gates

一次批次只有同时满足以下条件才可以进入人工审核完成后的发布步骤：

- `pending_analysis_windows == 0`
- `export_failures == 0`
- `segments_below_minimum == 0`
- 每条 source video 恰好有一条 summary
- timeline 全覆盖且无重叠
- `training_segments.jsonl` 中所有行无 quality flag
- episode ID 和 segment ID 全局唯一

建议持续监控：`uncovered_ratio`、`idle_ratio`、review/training 比例、各动作的数量与时长分布、
重试率、解析失败率和 `merged_for_min_duration` 比例。门禁失败时保留原始产物，不要直接放宽
阈值；先抽样定位 prompt、采样、窗口大小还是边界拼接的问题。
