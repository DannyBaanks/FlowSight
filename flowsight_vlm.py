"""FlowSight M4-C — Extensible contract & opt-in adapter for crop-only VLM escalation (`vlm_probe`).

DISABLED BY DEFAULT.
No network requests, no credentials, no external image transfers occur unless a
caller explicitly enables `vlm_probe` (via VLMProbePolicy, CLI `--enable-vlm`,
or `FLOWSIGHT_VLM_ENABLED=1`) AND supplies an authorized backend endpoint.

Epistemic & safety invariants:
  - Default policy: `enabled=False`, `allow_remote=False`, `require_crop_subregion=True`.
  - Refuses full-frame escalation when only a subregion crop is authorized.
  - Bounds crop dimensions (`<= 1024` per side, `<= 512*512` pixels) and timeouts.
  - Enforces loopback-only endpoints unless `allow_remote=True` is explicitly set.
  - Every VLM description is permanently tagged `provenance="VLM_INFERRED"` and
    bound to the exact `global_bbox` measured by FlowSight; it NEVER replaces or
    overwrites geometric/OCR measurements.
  - Never prints, returns, or persists API keys.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from typing import Any, Callable
from urllib.parse import urlparse
import urllib.request

from PIL import Image

VLM_DISABLED = 'VLM_DISABLED'
VLM_DENIED_POLICY = 'VLM_DENIED_POLICY'
VLM_UNAVAILABLE = 'VLM_UNAVAILABLE'
VLM_ERROR = 'VLM_ERROR'
VLM_INFERRED = 'VLM_INFERRED'

LOOPBACK_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})


@dataclass(frozen=True)
class VLMProbePolicy:
    enabled: bool = False
    allow_remote: bool = False
    require_crop_subregion: bool = True
    max_crop_dim: int = 1024
    max_crop_pixels: int = 512 * 512
    timeout_s: float = 15.0
    allowed_models: tuple[str, ...] = ()
    default_model: str = ''


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


def is_loopback_url(url: str) -> bool:
    """Return True only for http(s) URLs targeting localhost / loopback."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ('http', 'https'):
        return False
    host = (parsed.hostname or '').lower()
    return host in LOOPBACK_HOSTS


def _validate_endpoint_url(url: str, policy: VLMProbePolicy) -> str | None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        return f'invalid VLM endpoint URL scheme/host: {url!r}'
    if not policy.allow_remote and not is_loopback_url(url):
        return (
            f'refusing remote VLM endpoint {parsed.hostname!r} with allow_remote=False '
            '(set FLOWSIGHT_VLM_ALLOW_REMOTE=1 or --vlm-allow-remote to authorize remote transfer)'
        )
    return None


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
    effective_model = req.model or policy.default_model
    if policy.allowed_models and effective_model not in policy.allowed_models:
        return f'model {effective_model!r} is not in allowed_models {policy.allowed_models}'
    if not req.question.strip():
        return 'question must be non-empty'
    return None


def encode_crop_png_base64(crop_rgb_bytes: bytes, bbox: tuple[int, int, int, int]) -> str:
    """Encode raw RGB crop bytes into a bounded in-memory PNG base64 string."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    expected = w * h * 3
    if len(crop_rgb_bytes) != expected:
        raise ValueError(f'crop RGB buffer length {len(crop_rgb_bytes)} != {w}*{h}*3 ({expected})')
    img = Image.frombytes('RGB', (w, h), crop_rgb_bytes)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def make_openai_compatible_backend(
    url: str,
    *,
    api_key: str = '',
    default_model: str = '',
    max_tokens: int = 256,
) -> Callable[[bytes, VLMProbeRequest, VLMProbePolicy], str]:
    """Build a stdlib HTTP OpenAI-compatible Vision chat/completions backend for cropped regions."""
    endpoint = url.strip()

    def _call(crop_rgb_bytes: bytes, req: VLMProbeRequest, policy: VLMProbePolicy) -> str:
        url_err = _validate_endpoint_url(endpoint, policy)
        if url_err is not None:
            raise PermissionError(url_err)
        model = (req.model or default_model or policy.default_model).strip()
        if not model:
            raise ValueError('VLM model name is required (set FLOWSIGHT_VLM_MODEL or model=... in vlm_probe)')
        b64_png = encode_crop_png_base64(crop_rgb_bytes, req.global_bbox)
        payload = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': 0.0,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You inspect ONLY a small UI subregion crop for FlowSight. '
                        'Describe concisely what is visually shown (icon, symbol, state, text) '
                        'without inventing unseen context outside the crop.'
                    ),
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': req.question.strip()},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/png;base64,{b64_png}'},
                        },
                    ],
                },
            ],
        }
        headers = {'Content-Type': 'application/json'}
        if api_key.strip():
            headers['Authorization'] = f'Bearer {api_key.strip()}'
        http_req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(http_req, timeout=policy.timeout_s) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        data = json.loads(raw)
        choices = data.get('choices') or []
        if not choices or not isinstance(choices[0], dict):
            raise ValueError('VLM response contained no choices')
        msg = choices[0].get('message') or {}
        content = msg.get('content', '')
        if isinstance(content, list):
            parts = [str(p.get('text', '')) for p in content if isinstance(p, dict) and p.get('type') == 'text']
            content = '\n'.join(parts)
        text = str(content).strip()
        if not text:
            raise ValueError('VLM response content was empty')
        return text[:1200]

    return _call


def _env_truthy(val: str | None) -> bool:
    return (val or '').strip().lower() in ('1', 'true', 'yes', 'on')


def resolve_runtime_vlm_config(
    *,
    enable: bool | None = None,
    url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    allow_remote: bool | None = None,
    timeout_s: float | None = None,
    env: dict[str, str] | None = None,
) -> tuple[VLMProbePolicy, Callable[[bytes, VLMProbeRequest, VLMProbePolicy], str] | None]:
    """Resolve opt-in VLM policy + backend from explicit CLI args and/or environment variables.

    Stays disabled (`DEFAULT_POLICY, None`) unless explicitly enabled.
    Supported env vars:
      - FLOWSIGHT_VLM_ENABLED=1
      - FLOWSIGHT_VLM_URL=http://127.0.0.1:11434/v1/chat/completions
      - FLOWSIGHT_VLM_MODEL=...
      - FLOWSIGHT_VLM_API_KEY=...        (optional for local endpoints; never logged)
      - FLOWSIGHT_VLM_ALLOW_REMOTE=0|1   (default 0: loopback only)
      - FLOWSIGHT_VLM_TIMEOUT_S=15.0
      - FLOWSIGHT_VLM_ALLOWED_MODELS=m1,m2
    """
    src_env = os.environ if env is None else env
    is_enabled = enable if enable is not None else _env_truthy(src_env.get('FLOWSIGHT_VLM_ENABLED'))
    if not is_enabled:
        return DEFAULT_POLICY, None

    eff_url = (url if url is not None else src_env.get('FLOWSIGHT_VLM_URL', '')).strip()
    eff_model = (model if model is not None else src_env.get('FLOWSIGHT_VLM_MODEL', '')).strip()
    eff_key = api_key if api_key is not None else src_env.get('FLOWSIGHT_VLM_API_KEY', '')
    eff_remote = (
        allow_remote if allow_remote is not None
        else _env_truthy(src_env.get('FLOWSIGHT_VLM_ALLOW_REMOTE'))
    )
    raw_timeout = str(timeout_s) if timeout_s is not None else src_env.get('FLOWSIGHT_VLM_TIMEOUT_S', '15.0')
    try:
        eff_timeout = max(1.0, min(60.0, float(raw_timeout)))
    except ValueError:
        eff_timeout = 15.0

    allowed_raw = src_env.get('FLOWSIGHT_VLM_ALLOWED_MODELS', '').strip()
    allowed_models = tuple(m.strip() for m in allowed_raw.split(',') if m.strip())
    if not allowed_models and eff_model:
        allowed_models = (eff_model,)

    policy = VLMProbePolicy(
        enabled=True,
        allow_remote=eff_remote,
        require_crop_subregion=True,
        timeout_s=eff_timeout,
        allowed_models=allowed_models,
        default_model=eff_model,
    )
    if not eff_url:
        return policy, None
    backend = make_openai_compatible_backend(eff_url, api_key=eff_key, default_model=eff_model)
    return policy, backend


def execute_vlm_probe(
    crop_rgb_bytes: bytes,
    req: VLMProbeRequest,
    policy: VLMProbePolicy = DEFAULT_POLICY,
    backend: Callable[[bytes, VLMProbeRequest, VLMProbePolicy], str] | None = None,
) -> VLMProbeResult:
    """Fail-closed gatekeeper for optional VLM escalation over a single cropped region."""
    crop_sha = hashlib.sha256(crop_rgb_bytes).hexdigest()
    eff_model = req.model or policy.default_model
    if not policy.enabled:
        return VLMProbeResult(
            status=VLM_DISABLED,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=eff_model,
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
            model=eff_model,
            detail=err,
        )
    if backend is None:
        return VLMProbeResult(
            status=VLM_UNAVAILABLE,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=eff_model,
            detail='no authorized VLM backend URL wired (set FLOWSIGHT_VLM_URL / --vlm-url)',
        )
    try:
        desc = backend(crop_rgb_bytes, req, policy)
        return VLMProbeResult(
            status=VLM_INFERRED,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=eff_model,
            description=str(desc).strip(),
            detail='semantic inference over explicit subregion crop (NEVER replaces geometric truth)',
        )
    except PermissionError as e:
        return VLMProbeResult(
            status=VLM_DENIED_POLICY,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=eff_model,
            detail=str(e),
        )
    except Exception as e:
        return VLMProbeResult(
            status=VLM_ERROR,
            provenance=VLM_INFERRED,
            global_bbox=list(req.global_bbox),
            source_sha256=req.source_sha256,
            crop_sha256=crop_sha,
            model=eff_model,
            detail=f'VLM backend error: {type(e).__name__}: {e}',
        )
