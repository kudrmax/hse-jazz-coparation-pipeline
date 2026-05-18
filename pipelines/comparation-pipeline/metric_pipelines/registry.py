"""Registry of all corpus-level metric pipelines в порядке прогона."""
from __future__ import annotations

from metric_pipelines.bar_rhythm_jsd import BarRhythmJsdPipeline
from metric_pipelines.base import BaseCorpusMetricPipeline
from metric_pipelines.mgeval import MgevalPipeline
from metric_pipelines.plagiarism import PlagiarismPipeline


def all_corpus_pipelines() -> list[BaseCorpusMetricPipeline]:
    return [MgevalPipeline(), BarRhythmJsdPipeline(), PlagiarismPipeline()]
