#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import argparse
import pathlib
import textwrap
import logging
import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

# OpenAI SDK (chat completions)
from openai import OpenAI

# 선택적 로컬 전처리기 (llama_cpp)
LLAMA_AVAILABLE = False
try:
    if os.getenv("LLAMA_MODEL"):
        from llama_cpp import Llama  # type: ignore
        LLAMA_AVAILABLE = True
except Exception:
    LLAMA_AVAILABLE = False

# ----------------------------- 설정 ------------------------------

KOR_WORD = re.compile(r"[ㄱ-ㅎ가-힣]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MULTI_WS = re.compile(r"\s+")

TARGET_CATEGORIES = {"정치", "사회", "경제", "국제"}


@dataclass
class Config:
    input_path: str
    outdir: str = "./outputs"
    n_topics: int = 8
    topn_per_topic: int = 8
    topic_words_topk: int = 12
    random_state: int = 0

    # Vectorizer
    max_features: int = 60000
    max_df: float = 0.95
    min_df: int = 3

    # LLM
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt: str = "You are a concise Korean news analyst."
    temperature: float = 0.4
    max_tokens: int = 900

    # Cleaning
    min_content_chars: int = 300


# ---------------------------- 유틸리티 ----------------------------

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def slugify(text: str, maxlen: int = 64) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:maxlen] or "untitled"


# --------------------------- 전처리기 -----------------------------

class TextCleaner:
    def __init__(self, stopwords: Optional[set] = None) -> None:
        self.stopwords = stopwords or DEFAULT_STOPWORDS

    def clean(self, text: str) -> str:
        # URL 제거 및 공백 정규화 → 한글 단어만 남기는 경량 전처리
        text = URL_RE.sub(" ", str(text))
        tokens = KOR_WORD.findall(text)
        tokens = [t for t in tokens if len(t) > 1 and t not in self.stopwords]
        return " ".join(tokens)


class LlamaPreprocessor:
    """선택적 로컬 전처리. 환경변수 LLAMA_MODEL 설정 시 활성화.

    - 간단한 정규화/토큰화 과정을 수행해 입력 일관성을 높이는 목적
    - llama_cpp 사용 가능 시 내부 토큰화를 병행 (필요시에만 동작)
    """

    def __init__(self, model_path: Optional[str]) -> None:
        self.model_path = model_path
        self.engine = None
        if model_path and LLAMA_AVAILABLE:
            try:
                self.engine = Llama(model_path=model_path, n_ctx=2048, n_threads=1, n_gpu_layers=0)
            except Exception:
                self.engine = None

    def process(self, text: str) -> str:
        base = text.replace("\u200b", " ")  # zero-width space 제거
        base = MULTI_WS.sub(" ", base).strip()
        if self.engine is not None:
            try:
                _ = self.engine.tokenize(base.encode("utf-8"))
            except Exception:
                pass
        return base


# --------------------------- 데이터셋 ------------------------------

class NewsDataset:
    REQUIRED = {"title", "content"}

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @classmethod
    def load(cls, path: str) -> "NewsDataset":
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {p}")
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".json", ".jsonl"}:
            df = pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
        else:
            raise ValueError("CSV, JSON, JSONL 형식만 지원합니다.")
        return cls(df)

    def ensure_columns(self) -> None:
        missing = self.REQUIRED - set(self.df.columns)
        if missing:
            raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")
        defaults = {
            "provider": "unknown",
            "category": "기타",
            "date": pd.NaT,
            "url": "",
        }
        for k, v in defaults.items():
            if k not in self.df.columns:
                self.df[k] = v
        self.df["title"] = self.df["title"].astype(str)
        self.df["content"] = self.df["content"].astype(str)

    def filter_and_dedupe(self, min_chars: int) -> None:
        before = len(self.df)
        self.df = self.df.dropna(subset=["content"]).copy()
        self.df = self.df[self.df["content"].str.len() >= min_chars]
        if "category" in self.df.columns:
            mask = self.df["category"].isin(TARGET_CATEGORIES)
            self.df = self.df[mask | ~mask]
        self.df = self.df.drop_duplicates(subset=["title", "provider"]).reset_index(drop=True)
        logging.info("입력 기사: %d → 필터 후: %d", before, len(self.df))

    def add_clean_column(self, cleaner: TextCleaner) -> None:
        self.df["clean"] = (self.df["title"].fillna("") + " " + self.df["content"].fillna("")).map(cleaner.clean)


# ------------------------- 토픽 모델러 ----------------------------

class LdaTopicModeler:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.vectorizer: Optional[CountVectorizer] = None
        self.lda: Optional[LatentDirichletAllocation] = None

    def fit(self, texts: Sequence[str]) -> None:
        self.vectorizer = CountVectorizer(
            max_df=self.cfg.max_df,
            min_df=self.cfg.min_df,
            max_features=self.cfg.max_features,
            token_pattern=r"(?u)[A-Za-z0-9가-힣]{2,}",
        )
        X = self.vectorizer.fit_transform(texts)
        self.lda = LatentDirichletAllocation(
            n_components=self.cfg.n_topics,
            learning_method="batch",
            max_iter=20,
            random_state=self.cfg.random_state,
            evaluate_every=0,
        )
        self.lda.fit(X)
        self._X = X

    def topic_words(self, topk: Optional[int] = None) -> List[str]:
        assert self.lda is not None and self.vectorizer is not None
        topk = topk or self.cfg.topic_words_topk
        fnames = np.array(self.vectorizer.get_feature_names_out())
        topics = []
        for comp in self.lda.components_:
            idx = np.argsort(comp)[-topk:][::-1]
            topics.append(", ".join(fnames[idx]))
        return topics

    def doc_topic_distribution(self) -> np.ndarray:
        assert self.lda is not None and self.vectorizer is not None
        theta = self.lda.transform(self._X)
        return normalize(theta, norm="l1", axis=1)

    def top_docs_for_topic(self, df: pd.DataFrame, theta: np.ndarray, topic_id: int, topn: int) -> pd.DataFrame:
        tmp = df.copy()
        tmp["topic_prob"] = theta[:, topic_id]
        return tmp.sort_values("topic_prob", ascending=False).head(topn)


# ----------------------------- 요약 -------------------------------

class ChatSummarizer:
    def __init__(self, cfg: Config, preprocessor: Optional[LlamaPreprocessor] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai>=1.0.0 패키지가 필요합니다. 'pip install openai' 후 재시도")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
        self.client = OpenAI(api_key=api_key)
        self.cfg = cfg
        self.pre = preprocessor

    def _build_prompt(self, topic_words: str, rows: pd.DataFrame) -> str:
        bullets: List[str] = []
        for _, r in rows.iterrows():
            snippet = r["content"][:600].replace("\n", " ")
            bullets.append(f"- [{r.get('provider','?')}] {r['title']}\n  {snippet}…")
        body = "\n".join(bullets)
        return textwrap.dedent(
            f"""
            다음은 동일 주제로 분류된 기사 묶음입니다. 핵심 내용을 간결하게 정리하세요.
            요구사항:
            - 공통 핵심 내용을 3~5문장으로 요약
            - 서로 다른 관점/강조점이 있으면 2~4개 불릿으로 비교
            - 독자가 알아두면 좋은 배경 맥락 1~2개 제시

            [주제 키워드]
            {topic_words}

            [기사 묶음]
            {body}

            형식: 한국어, 담백한 문체, 과장/추측/감정 표현 지양, 사실 위주.
            """
        ).strip()

    def summarize(self, topic_words: str, rows: pd.DataFrame) -> str:
        prompt = self._build_prompt(topic_words, rows)
        if self.pre is not None:
            prompt = self.pre.process(prompt)
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=[
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logging.warning("요약 호출 실패: %s", e)
            return textwrap.shorten(prompt, width=1200, placeholder="…")


# -------------------------- 결과 저장소 ----------------------------

class ResultWriter:
    def __init__(self, outdir: str) -> None:
        self.outdir = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def write_topic_markdown(self, topic_id: int, keywords: str, reps: pd.DataFrame, summary: str) -> pathlib.Path:
        md = []
        md.append(f"# 토픽 {topic_id:02d} — 키워드\n\n{keywords}\n")
        md.append(f"## 대표 기사 ({len(reps)}개)\n")
        for r in reps.itertuples():
            line = f"- [{getattr(r, 'provider', 'unknown')}] {r.title}"
            if getattr(r, 'url', ''):
                line += f" — {r.url}"
            md.append(line)
        md.append("\n## 요약\n\n" + summary + "\n")
        path = self.outdir / f"topic_{topic_id:02d}.md"
        path.write_text("\n".join(md), encoding="utf-8")
        return path

    def write_index(self, records: List[Dict[str, Any]]) -> pathlib.Path:
        p = self.outdir / "index_topics.json"
        p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def write_overview_markdown(self, records: List[Dict[str, Any]]) -> pathlib.Path:
        md = ["# 토픽 개요\n"]
        for r in records:
            md.append(f"- 토픽 {r['topic_id']:02d}: {r['keywords']} ({r['n_articles']}개) → {r['markdown']}")
        p = self.outdir / "overview.md"
        p.write_text("\n".join(md) + "\n", encoding="utf-8")
        return p


# --------------------------- 파이프라인 ---------------------------

def run_pipeline(cfg: Config) -> Dict[str, Any]:
    setup_logging()
    seed_everything(cfg.random_state)

    # 1) 로드
    ds = NewsDataset.load(cfg.input_path)
    ds.ensure_columns()
    ds.filter_and_dedupe(min_chars=cfg.min_content_chars)

    # 2) 정제 텍스트 생성
    cleaner = TextCleaner()
    ds.add_clean_column(cleaner)

    # 3) 토픽 모델 학습
    modeler = LdaTopicModeler(cfg)
    modeler.fit(ds.df["clean"].tolist())
    topics = modeler.topic_words(cfg.topic_words_topk)
    theta = modeler.doc_topic_distribution()

    # 4) 요약기 준비 (선택적 전처리 포함)
    llama_pre = LlamaPreprocessor(os.getenv("LLAMA_MODEL")) if os.getenv("LLAMA_MODEL") else None
    summarizer = ChatSummarizer(cfg, preprocessor=llama_pre)

    # 5) 토픽별 대표 기사 → 요약 → 파일 저장
    writer = ResultWriter(cfg.outdir)
    records: List[Dict[str, Any]] = []

    base_df = ds.df[["title", "content", "provider", "url"]].copy()

    for k, words in enumerate(topics):
        reps = modeler.top_docs_for_topic(base_df, theta, topic_id=k, topn=cfg.topn_per_topic)
        summary = summarizer.summarize(words, reps)
        md_path = writer.write_topic_markdown(k, words, reps, summary)
        records.append({
            "topic_id": k,
            "keywords": words,
            "n_articles": int(len(reps)),
            "markdown": str(md_path),
        })

    index_path = writer.write_index(records)
    overview_path = writer.write_overview_markdown(records)

    logging.info("완료: 토픽 %d개, 인덱스=%s, 개요=%s", len(records), index_path, overview_path)
    return {
        "n_topics": cfg.n_topics,
        "outputs": records,
        "index": str(index_path),
        "overview": str(overview_path),
    }


# --------------------------- CLI 진입점 ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LDA 기반 뉴스 토픽 요약 (단일 LLM 요약)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", dest="input_path", required=True, help="입력 파일 경로(CSV/JSON/JSONL)")
    p.add_argument("--outdir", default="./outputs", help="출력 폴더")
    p.add_argument("--n-topics", dest="n_topics", type=int, default=8, help="LDA 토픽 개수")
    p.add_argument("--topn-per-topic", dest="topn_per_topic", type=int, default=8, help="토픽별 대표 기사 수")
    p.add_argument("--topic-words-topk", dest="topic_words_topk", type=int, default=12, help="토픽 키워드 상위 k")
    p.add_argument("--max-features", dest="max_features", type=int, default=60000, help="벡터라이저 최대 피처 수")
    p.add_argument("--max-df", dest="max_df", type=float, default=0.95, help="벡터라이저 max_df")
    p.add_argument("--min-df", dest="min_df", type=int, default=3, help="벡터라이저 min_df")
    p.add_argument("--min-content-chars", dest="min_content_chars", type=int, default=300, help="기사 본문 최소 길이")
    p.add_argument("--random-state", dest="random_state", type=int, default=0, help="랜덤 시드")
    p.add_argument("--openai-model", dest="openai_model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI 모델명")
    p.add_argument("--temperature", dest="temperature", type=float, default=0.4, help="요약 생성 온도")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=900, help="요약 최대 토큰 수")
    return p


def parse_args_to_config(args: argparse.Namespace) -> Config:
    return Config(
        input_path=args.input_path,
        outdir=args.outdir,
        n_topics=args.n_topics,
        topn_per_topic=args.topn_per_topic,
        topic_words_topk=args.topic_words_topk,
        random_state=args.random_state,
        max_features=args.max_features,
        max_df=args.max_df,
        min_df=args.min_df,
        openai_model=args.openai_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_content_chars=args.min_content_chars,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = parse_args_to_config(args)

    info = run_pipeline(cfg)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
