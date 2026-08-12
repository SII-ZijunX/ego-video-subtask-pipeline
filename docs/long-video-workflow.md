# Long-video workflow

## 1. Selection

`prepare` 从合法 Ego4D mount 中选择指定时长范围的 clip。选择由 seed 和 clip UID 的稳定
hash 决定，并优先保证 scenario 多样性。narration 不参与选择。

## 2. Analysis windows

长视频按 `window_sec` 切成重叠窗口，最后一个窗口贴齐视频结尾，确保 `[0, duration]` 无空洞。
窗口会缩放并转码成浏览器兼容的 H.264 MP4，音频默认移除。它们只用于推理。

## 3. Window annotation

VLM 接收按时间采样的帧、明确的时间戳和 JSON schema，输出视频级 caption 与多个局部
subtask。系统保留 prompt、raw response、解析和 repair/validation 结果。失败可重试，batch
可断点续跑。

## 4. Global reconciliation

局部起止时间加上窗口 offset 后映射回 source time。重叠窗口内的候选按以下优先级选择：

1. 可训练候选优先；
2. 非窗口边缘候选优先；
3. 离窗口边缘更远者优先；
4. 较长候选优先；
5. 稳定 ID 作为最后 tie-breaker。

同 action/object 的相邻原子区间合并。没有任何候选覆盖的区域不会删除，而是生成
`uncovered_by_qwen` review-only filler。

## 5. Three-second hard minimum

短于 3 秒的区间反复与相邻段合并，直到所有最终区间满足硬下限。优先合并相同语义；若只能
与不同动作合并，输出变成 `fine_action=compound`，增加 `merged_for_min_duration`，并设置
`training_eligible=false`。这样不会为了时长门禁伪造单一动作 label。

## 6. Export and review

最终边界确定后才从 source video 导出 MP4。系统用 ffprobe 核验实际时长，并生成：完整时间线、
训练候选、review queue、每视频 summary、统计和 HTML 审查页。

`review_decisions.jsonl` 是人工决策模板，允许记录 boundary/label 正确性、修正时间和修正 label。
当前版本不会自动把人工修改写回训练 manifest；建议通过单独、可审计的审核合并步骤完成。
