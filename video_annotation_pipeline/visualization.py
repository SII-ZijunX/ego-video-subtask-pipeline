"""Self-contained local HTML review pages with synchronized video controls."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .schemas import EpisodeAnnotation, EpisodeMetadata


ACTION_COLORS = {
    "move": "#4e79a7", "fold": "#f28e2b", "pour": "#e15759", "unfold": "#76b7b2",
    "push": "#59a14f", "wipe": "#edc948", "pull": "#b07aa1", "stir": "#ff9da7",
    "rotate": "#9c755f", "cut": "#bab0ab", "open": "#2f8f9d", "press": "#8d6e63",
    "close": "#6a5acd", "attach": "#00a878", "detach": "#d1495b", "transit": "#6c757d",
    "idle": "#adb5bd", "other": "#343a40",
}


def generate_episode_report(
    annotation: EpisodeAnnotation,
    metadata: EpisodeMetadata,
    episode_dir: Path,
    output_path: Path,
    raw_response: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    videos = []
    for camera in metadata.cameras:
        source = metadata.resolved_camera_path(episode_dir, camera).resolve()
        relative = os.path.relpath(source, output_path.parent)
        videos.append(
            f'<section><h3>{html.escape(camera.role)} · {html.escape(camera.name)}</h3>'
            f'<video controls preload="metadata" data-offset="{camera.time_offset_sec}" src="{html.escape(relative)}"></video></section>'
        )
    segments = []
    for subtask in annotation.subtasks:
        left = 100 * subtask.start_time_sec / annotation.duration_sec
        width = 100 * (subtask.end_time_sec - subtask.start_time_sec) / annotation.duration_sec
        color = ACTION_COLORS.get(subtask.action, ACTION_COLORS["other"])
        payload = html.escape(json.dumps(subtask.model_dump(mode="json"), ensure_ascii=False))
        segments.append(
            f'<button class="segment" style="left:{left:.3f}%;width:{max(width, 0.5):.3f}%;background:{color}" '
            f'data-start="{subtask.start_time_sec}" data-detail="{payload}">{html.escape(subtask.action)}</button>'
        )
    flags = "".join(f"<li>{html.escape(flag)}</li>" for flag in annotation.episode_quality_flags) or "<li>None</li>"
    reference_caption = str(getattr(metadata, "reference_caption", "") or "")
    reference_block = (
        f'<p><b>Official reference:</b> {html.escape(reference_caption)}</p>'
        if reference_caption else ""
    )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{action}</span>' for action, color in ACTION_COLORS.items()
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(annotation.episode_id)} review</title>
<style>
body{{font:15px system-ui;margin:24px;max-width:1280px;background:#f7f7f8;color:#202124}} video{{width:100%;max-height:420px;background:#111}} .videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}} section,.card{{background:white;padding:16px;border-radius:10px;box-shadow:0 1px 5px #0002}} .timeline{{position:relative;height:48px;background:#ddd;margin:16px 0;border-radius:6px;overflow:hidden}} .segment{{position:absolute;top:0;height:100%;border:0;color:white;overflow:hidden;cursor:pointer;border-right:1px solid #fff}} .legend span{{display:inline-flex;align-items:center;margin:3px 10px 3px 0}} .legend i{{width:12px;height:12px;margin-right:4px}} pre{{white-space:pre-wrap;max-height:360px;overflow:auto;background:#111;color:#eee;padding:12px}} .review button{{padding:10px 16px;margin-right:8px}} #detail{{min-height:70px}}
</style></head><body>
<h1>{html.escape(annotation.episode_id)}</h1>
<div class="card"><h2>{html.escape(annotation.video_level_instruction)}</h2>{reference_block}<div class="timeline">{''.join(segments)}</div><div class="legend">{legend}</div><pre id="detail">Click a subtask to inspect and seek all videos.</pre><h3>Quality flags</h3><ul>{flags}</ul><div class="review"><b>Human review:</b> <button data-review="accept">accept</button><button data-review="reject">reject</button><button data-review="needs_edit">needs_edit</button> <span id="review-state"></span><br><textarea id="review-notes" rows="3" cols="70" placeholder="Optional review notes"></textarea><br><button id="export-review">Export review JSON</button></div></div>
<h2>Videos</h2><div class="videos">{''.join(videos)}</div>
<div class="card"><h2>Raw VLM response</h2><pre>{html.escape(raw_response)}</pre></div>
<script>
const videos=[...document.querySelectorAll('video')];
document.querySelectorAll('.segment').forEach(button=>button.onclick=()=>{{const t=Number(button.dataset.start);videos.forEach(v=>v.currentTime=Math.max(0,t+Number(v.dataset.offset||0)));document.querySelector('#detail').textContent=JSON.stringify(JSON.parse(button.dataset.detail),null,2);}});
videos.forEach((video,index)=>{{video.addEventListener('play',()=>{{const globalTime=video.currentTime-Number(video.dataset.offset||0);videos.forEach((v,i)=>{{if(i!==index){{v.currentTime=Math.max(0,globalTime+Number(v.dataset.offset||0));v.play();}}}});}});video.addEventListener('pause',()=>videos.forEach((v,i)=>{{if(i!==index)v.pause();}}));}});
const key='review:{html.escape(annotation.episode_id)}'; const state=document.querySelector('#review-state'); state.textContent=localStorage.getItem(key)||'unreviewed'; document.querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>{{localStorage.setItem(key,b.dataset.review);state.textContent=b.dataset.review;}});
document.querySelector('#export-review').onclick=()=>{{const payload={{episode_id:{json.dumps(annotation.episode_id)},status:localStorage.getItem(key)||'unreviewed',notes:document.querySelector('#review-notes').value,reviewed_at:new Date().toISOString()}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download={json.dumps(annotation.episode_id + '.review.json')};a.click();URL.revokeObjectURL(a.href);}};
</script></body></html>"""
    output_path.write_text(document)
    return output_path
