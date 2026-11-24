"""FlowSight M4-C — Extensible contract for optional crop-only VLM escalation (`vlm_probe`).

DISABLED BY DEFAULT.
No network requests, no credentials, no external image transfers occur unless a
caller explicitly constructs an authorized policy AND supplies a verified backend.

Epistemic & safety invariants:
  - Default policy: `enabled=False`, `allow_remote=False`, `require_crop_subregion=True`.
  - Refuses full-frame escalation when only a subregion crop is authorized.
  - Bounds crop dimensions (`<= 1024` per side, `<= 512*512` pixels) and timeouts.
  - Every VLM description is permanently tagged `provenance="VLM_INFERRED"` and
    bound to the exact `global_bbox` measured by FlowSight; it NEVER replaces or
    overwrites geometric/OCR measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable

VLM_DISABLED = 'VLM_DISABLED'
VLM_DENIED_POLICY = 'VLM_DENIED_POLICY'
VLM_UNAVAILABLE = 'VLM_UNAVAILABLE'
VLM_ERROR = 'VLM_ERROR'
VLM_INFERRED = 'VLM_INFERRED'


@dataclass(frozen=True)
class VLMProbePolicy:
    enabled: bool = False
    allow_remote: bool = False
    require_crop_subregion: bool = True
    max_crop_dim: int = 1024
    max_crop_pixels: int = 512 * 512
    timeout_s: float = 15.0
    allowed_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class VLMProbeRequest:
    source_sha256: str
    full_size: tuple[int, int]
    global_bbox: tuple[int, int, int, int]
    crop_sha256: str
    question: str
    model: str = ''
    reason: str = 'ambiguous_region'


@dataclass
class VLMProbeResult:
    status: str
    provenance: str
    global_bbox: list[int]
    source_sha256: str
    crop_sha256: str
    model: str
    description: str | None = None
    detail: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'provenance': self.provenance,
            'global_bbox': self.global_bbox,
            'source_sha256': self.source_sha256,
            'crop_sha256': self.crop_sha256,
            'model': self.model,
            'description': self.description,
            'detail': self.detail,
            'metadata': self.metadata,
        }


DEFAULT_POLICY = VLMProbePolicy()


def validate_vlm_request(req: VLMProbeRequest, policy: VLMProbePolicy) -> str | None:
    """Validate policy constraints before any pixel data leaves FlowSight. Returns error reason or None."""
    if not policy.enabled:
        return 'VLM escalation is disabled by default (set explicit authorized policy to enable)'
    fw, fh = req.full_size
    x0, y0, x1, y1 = req.global_bbox
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0 or x0 < 0 or y0 < 0 or x1 > fw or y1 > fh:
        return f'invalid global_bbox {req.global_bbox} for full_size {req.full_size}'
    if policy.require_crop_subregion and (bw >= fw and bh >= fh):
        return 'refusing full-frame VLM transfer; apply crop=x:y:w:h to select only the ambiguous subregion'
    if bw > policy.max_crop_dim or bh > policy.max_crop_dim or (bw * bh) > policy.max_crop_pixels:
        return f'crop {bw}x{bh} exceeds policy limit ({policy.max_crop_dim}px/side, {policy.max_crop_pixels}px total)'
    if policy.allowed_models and req.model not in policy.allowed_models:
        return f'model {req.model!r} is not in allowed_models {policy.allowed_models}'
    if not req.question.strip():
        return 'question must be non-empty'
    return None


def execute_vlm_probe(
    crop_rgb_bytes: bytes,
    req: VLMProbeRequest,
    policy: VLMProbePolicy = DEFAULT_POLICY,
    backend: Callable[[bytes, VLMProbeRequest, VLMProbePolicy], str] | None = None,
) -> VLMProbeResult:
    """Fail-closed gatekeeper for optional VLM escalation over a single cropped region."""
    crop_sha = hashlib.sha256(crop_rgb_bytes).hexdigest()
    if not policy.enabled:
        return VLMProbeResult(
            status=VLM_DISABLED,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=req.model,
            detail='vlm_probe disabled by default; geometric/OCR evidence preserved untouched',
        )
    err = validate_vlm_request(req, policy)
    if err is not None:
        return VLMProbeResult(
            status=VLM_DENIED_POLICY,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=req.model,
            detail=err,
        )
    if backend is None:
        return VLMProbeResult(
            status=VLM_UNAVAILABLE,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=req.model,
            detail='no authorized VLM backend wired (M4-C contract only; no implicit network call)',
        )
    try:
        desc = backend(crop_rgb_bytes, req, policy)
        return VLMProbeResult(
            status=VLM_INFERRED,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=req.model,
            description=str(desc).strip(),
            detail='semantic inference over explicit subregion crop (NEVER replaces geometric truth)',
        )
    except Exception as e:
        return VLMProbeResult(
            status=VLM_ERROR,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=req.model,
            detail=f'VLM backend error: {type(e).__name__}: {e}',
        )
