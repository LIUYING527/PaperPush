#!/usr/bin/env python3
"""
PaperPush daily arXiv paper fetcher.
Fetches top papers from cs.AI/LG/CL/CV/RO and generates bilingual summaries
via DeepSeek API. Outputs data/YYYY-MM-DD.json for the PWA to consume.
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date

import requests

# ── Config ─────────────────────────────────────────────────────────────────
CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.RO']
PAPERS_PER_CATEGORY = 4   # Fetch extra to account for cross-posting dedup
TARGET_TOTAL = 10         # Final paper count

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL = 'https://api.deepseek.com/v1/chat/completions'
ARXIV_API_URL = 'http://export.arxiv.org/api/query'

# arXiv Atom XML namespaces
NS = {
    'atom':   'http://www.w3.org/2005/Atom',
    'arxiv':  'http://arxiv.org/schemas/atom',
}


# ── arXiv fetching ──────────────────────────────────────────────────────────
def fetch_arxiv(category: str, max_results: int = 5) -> list[dict]:
    """Fetch the most recently submitted papers in a category."""
    params = {
        'search_query': f'cat:{category}',
        'sortBy':       'submittedDate',
        'sortOrder':    'descending',
        'start':        0,
        'max_results':  max_results,
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall('atom:entry', NS):
        # ID: strip version suffix (e.g. "2506.12345v1" → "2506.12345")
        raw_id = entry.find('atom:id', NS).text
        arxiv_id = raw_id.split('/abs/')[-1].split('v')[0]

        title = ' '.join(
            entry.find('atom:title', NS).text.strip().split()
        )  # normalise whitespace

        authors = [
            a.find('atom:name', NS).text
            for a in entry.findall('atom:author', NS)
        ]

        abstract = ' '.join(
            entry.find('atom:summary', NS).text.strip().split()
        )

        published = entry.find('atom:published', NS).text  # ISO 8601

        cats = [c.get('term') for c in entry.findall('atom:category', NS)]
        pc_el = entry.find('arxiv:primary_category', NS)
        primary = pc_el.get('term') if pc_el is not None else (cats[0] if cats else category)

        papers.append({
            'id':               arxiv_id,
            'title':            title,
            'authors':          authors,
            'abstract':         abstract,
            'published':        published,
            'primary_category': primary,
            'categories':       cats,
            'arxiv_url':        f'https://arxiv.org/abs/{arxiv_id}',
            'pdf_url':          f'https://arxiv.org/pdf/{arxiv_id}',
        })

    return papers


# ── DeepSeek summarisation ──────────────────────────────────────────────────
SUMMARY_PROMPT = """\
你是一位AI/ML领域的论文研究助手，擅长用简洁语言向研究人员解释论文核心。

论文标题: {title}
论文摘要: {abstract}

请严格按下面的JSON格式输出，不要有任何额外内容或markdown代码块:
{{"summary_zh":"中文三句话：第一句讲问题/背景，第二句讲方法/创新，第三句讲结果/意义","summary_en":"English 3 sentences: background, method/novelty, results/impact","highlights_zh":["核心创新点（≤20字）","最佳实验结果（≤20字）","应用价值或局限（≤20字）"]}}"""


def summarize(paper: dict) -> dict:
    """Call DeepSeek to produce bilingual summary + 3 highlights."""
    if not DEEPSEEK_API_KEY:
        return {
            'summary_zh':    '（需配置 DEEPSEEK_API_KEY）',
            'summary_en':    '(DEEPSEEK_API_KEY not set)',
            'highlights_zh': ['暂无', '暂无', '暂无'],
        }

    payload = {
        'model':       'deepseek-chat',
        'temperature': 0.3,
        'max_tokens':  700,
        'messages':    [{
            'role':    'user',
            'content': SUMMARY_PROMPT.format(
                title=paper['title'],
                abstract=paper['abstract'][:2000],   # keep cost bounded
            ),
        }],
    }
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type':  'application/json',
    }

    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    raw = resp.json()['choices'][0]['message']['content'].strip()
    # Strip accidental markdown fences
    raw = re.sub(r'^```(?:json)?|```$', '', raw, flags=re.MULTILINE).strip()

    return json.loads(raw)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    today_str = date.today().strftime('%Y-%m-%d')
    print(f'[PaperPush] Fetching papers for {today_str}')

    # 1. Collect papers, deduplicate by arXiv ID
    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for cat in CATEGORIES:
        print(f'  → {cat}')
        try:
            batch = fetch_arxiv(cat, max_results=PAPERS_PER_CATEGORY)
        except Exception as exc:
            print(f'    ERROR fetching {cat}: {exc}')
            batch = []

        for p in batch:
            if p['id'] not in seen_ids:
                seen_ids.add(p['id'])
                p['_source_cat'] = cat   # which query surfaced it first
                all_papers.append(p)

        time.sleep(3)   # polite delay for arXiv

    # 2. Balanced selection: ~2 per category, up to TARGET_TOTAL
    selected: list[dict] = []
    per_cat: dict[str, int] = {c: 0 for c in CATEGORIES}

    for p in all_papers:
        if len(selected) >= TARGET_TOTAL:
            break
        cat = p['_source_cat']
        if per_cat[cat] < 2:
            selected.append(p)
            per_cat[cat] += 1

    # Fill any remaining slots from overflow
    for p in all_papers:
        if p not in selected and len(selected) < TARGET_TOTAL:
            selected.append(p)

    # 3. Generate summaries
    print(f'\n[PaperPush] Summarising {len(selected)} papers via DeepSeek…')
    for i, paper in enumerate(selected):
        print(f'  [{i+1}/{len(selected)}] {paper["title"][:70]}')
        try:
            summary = summarize(paper)
            paper.update(summary)
        except Exception as exc:
            print(f'    WARN summarise failed: {exc}')
            paper.update({
                'summary_zh':    '摘要生成失败',
                'summary_en':    'Summary generation failed',
                'highlights_zh': ['—', '—', '—'],
            })
        # Clean up internal key
        paper.pop('_source_cat', None)
        time.sleep(1)

    # 4. Write output JSON
    os.makedirs('data', exist_ok=True)
    output = {
        'date':         today_str,
        'categories':   CATEGORIES,
        'count':        len(selected),
        'papers':       selected,
    }
    out_path = f'data/{today_str}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\n[PaperPush] Done. {len(selected)} papers → {out_path}')


if __name__ == '__main__':
    main()
