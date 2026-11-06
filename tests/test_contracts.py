"""Synthetic software contracts; no recorded speech or model downloads."""
from pathlib import Path
import numpy as np
import pytest
from src.io.cache import get_transcript_path, load_cached_transcript, save_transcript
from src.io.dataset import load_manifest
from src.features.graph_semantic import compute_s_sem

def test_cache_round_trip_and_input_separation(tmp_path):
    fixture = {"text": "Fictional description", "words": []}
    assert load_cached_transcript(tmp_path, 'example-a', 'one.wav') is None
    save_transcript(tmp_path, 'example-a', 'one.wav', fixture)
    assert load_cached_transcript(tmp_path, 'example-a', 'one.wav') == fixture
    assert get_transcript_path(tmp_path, 'example-a', 'one.wav') != get_transcript_path(tmp_path, 'example-a', 'two.wav')
    assert get_transcript_path(tmp_path, 'example-a', 'one.wav') != get_transcript_path(tmp_path, 'example-b', 'one.wav')

def test_corrupted_cache_is_a_miss(tmp_path):
    path = save_transcript(tmp_path, 'example', 'one.wav', {})
    path.write_text('{broken')
    assert load_cached_transcript(tmp_path, 'example', 'one.wav') is None

def test_manifest_requires_input_identity_and_resolves_relative_audio(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('subject_id,audio_path\nexample,one.wav\n')
    frame = load_manifest(manifest)
    assert frame.iloc[0].audio_path_resolved == (tmp_path / 'one.wav').resolve()
    manifest.write_text('subject_id\nexample\n')
    with pytest.raises(ValueError, match='missing columns'):
        load_manifest(manifest)

def test_graph_evidence_for_fictional_description():
    brief, _ = compute_s_sem({"text": "A man is here."})
    detailed, details = compute_s_sem({"text": "The man is reading a book on the blanket. The boy is flying a kite. The dog is chasing the girl near the lake."})
    assert 0 <= brief < detailed <= 1
    assert details['matched_cu_count'] > 0
    assert 'boy' in details['matched_nodes']

def test_missing_reference_returns_missing_score(tmp_path):
    score, details = compute_s_sem({"text": "Fictional text"}, tmp_path / 'absent.json')
    assert np.isnan(score) and details == {}
