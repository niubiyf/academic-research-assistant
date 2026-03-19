"""
完整管道：输入研究主题 → 搜索论文 → 提取摘要 → 大模型分析 → 多文献归纳

=== 这个文件做了什么？===

把四个模块串成一条流水线（对应题目要求的完整流程）：

  用户输入主题（如 "transformer attention mechanism"）
       ↓
  ① arxiv_searcher         →  搜索 arXiv，下载 PDF
       ↓
  ② pdf_processor          →  从 PDF 提取摘要（兜底用）
       ↓
  ③ paper_analyzer         →  大模型逐篇分析（Map 阶段）
       ↓
  ④ multi_paper_summarizer →  多文献归纳（Reduce 阶段）
       ↓
  输出：主流方法 + 优缺点 + 调研报告大纲
"""

import os
import sys
import json
import time

sys.path.append(os.path.dirname(__file__))

from modules.arxiv_searcher import search_papers, download_pdf
from modules.pdf_processor import process_pdf
from modules.paper_analyzer import PaperAnalyzer
from modules.multi_paper_summarizer import MultiPaperSummarizer


# ============================================================
# 配置区域（从环境变量或 .streamlit/secrets.toml 读取，不硬编码密钥）
# ============================================================

from config_loader import load_api_config

_CONFIG = load_api_config()
API_KEY = _CONFIG["api_key"]
BASE_URL = _CONFIG["base_url"]
MODEL = _CONFIG["model"]

# PDF 保存目录
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")


# ============================================================
# 核心管道
# ============================================================

def run_pipeline(query: str, max_papers: int = 5) -> dict:
    """
    完整管道：从搜索到分析一步到位

    === 流程 ===
    Step 1: 搜索 arXiv
    Step 2: 下载 PDF
    Step 3: 提取摘要（两种来源：PDF 提取 或 arXiv 自带摘要）
    Step 4: 大模型逐篇分析（Map 阶段）
    Step 5: 多文献归纳（Reduce 阶段）→ 主流方法 + 优缺点 + 报告大纲
    Step 6: 汇总并保存

    Args:
        query:      搜索关键词，如 "federated learning privacy"
        max_papers: 最多分析几篇（建议 3-5 篇，省钱省时间）

    Returns:
        包含所有分析结果的字典
    """

    results = {
        "query": query,
        "papers": [],
        "summary": {},
    }

    # ============================================================
    # Step 1: 搜索 arXiv
    # ============================================================
    print("=" * 60)
    print(f"Step 1: 搜索 arXiv — \"{query}\"")
    print("=" * 60)

    papers = search_papers(query, max_results=max_papers)

    if not papers:
        print("未找到论文，请换个关键词试试")
        return results

    # ============================================================
    # Step 2: 下载 PDF
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"Step 2: 下载 PDF ({len(papers)} 篇)")
    print("=" * 60)

    for i, paper in enumerate(papers):
        pdf_path = download_pdf(paper, save_dir=PAPERS_DIR)
        paper["local_pdf_path"] = pdf_path

        if i < len(papers) - 1:
            time.sleep(3)  # arXiv 速率限制

    # ============================================================
    # Step 3: 提取摘要
    # ============================================================
    #
    # 这里有个巧妙的设计：我们有两个摘要来源
    #
    # 来源 A：arXiv API 返回的摘要（search_papers 已经拿到了）
    # 来源 B：从 PDF 里用 pdf_processor 提取的摘要
    #
    # 优先用来源 A（更干净、不需要解析 PDF）
    # 如果来源 A 太短或为空，才用来源 B 做兜底
    #
    # 这就是"多来源容错"的思路，和你 pdf_processor 里
    # "先 PyMuPDF 后 pdfplumber"的策略是一样的道理

    print(f"\n{'=' * 60}")
    print("Step 3: 准备摘要")
    print("=" * 60)

    for paper in papers:
        # 优先使用 arXiv 自带摘要
        arxiv_abstract = paper.get("abstract", "")

        if len(arxiv_abstract) > 100:
            paper["final_abstract"] = arxiv_abstract
            paper["abstract_source"] = "arXiv API"
            print(f"  [{paper['arxiv_id']}] 使用 arXiv 摘要 ({len(arxiv_abstract)} 字符)")
        else:
            # 兜底：从 PDF 提取
            pdf_path = paper.get("local_pdf_path")
            if pdf_path and os.path.exists(pdf_path):
                pdf_data = process_pdf(pdf_path)
                paper["final_abstract"] = pdf_data.get("abstract", arxiv_abstract)
                paper["abstract_source"] = "PDF 提取"
                print(f"  [{paper['arxiv_id']}] 使用 PDF 提取摘要")
            else:
                paper["final_abstract"] = arxiv_abstract
                paper["abstract_source"] = "arXiv API (不完整)"
                print(f"  [{paper['arxiv_id']}] 警告：摘要可能不完整")

    # ============================================================
    # Step 4: 大模型逐篇分析
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"Step 4: 大模型分析 ({len(papers)} 篇)")
    print("=" * 60)

    analyzer = PaperAnalyzer(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    for i, paper in enumerate(papers):
        print(f"\n--- 论文 {i+1}/{len(papers)} ---")

        analysis = analyzer.analyze(
            title=paper["title"],
            abstract=paper["final_abstract"]
        )

        # 把分析结果合并到论文字典里
        paper["analysis"] = analysis

        # 收集到 results 中
        results["papers"].append({
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "abstract_source": paper["abstract_source"],
            "analysis": analysis,
        })

        # API 调用间隔（避免触发限流）
        if i < len(papers) - 1:
            time.sleep(1)

    # ============================================================
    # Step 5: 多文献归纳（Reduce 阶段）
    # ============================================================
    #
    # 这是题目要求的核心功能：
    # - "研究观点归纳（如总结出该领域的三个主流方法及其优缺点）"
    # - "生成调研报告大纲"
    #
    # 原理：把 Step 4 每篇论文的分析结果（已经是精简的 JSON），
    # 打包发给大模型，让它跨论文归纳出共性和差异

    print(f"\n{'=' * 60}")
    print("Step 5: 多文献归纳（研究观点 + 优缺点 + 报告大纲）")
    print("=" * 60)

    summarizer = MultiPaperSummarizer(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL
    )

    summary = summarizer.summarize(results["papers"])
    results["summary"] = summary

    # 打印归纳结果
    print("\n" + summarizer.format_result(summary))

    # ============================================================
    # Step 6: 汇总并保存
    # ============================================================
    print(f"{'=' * 60}")
    print("Step 6: 保存所有结果")
    print("=" * 60)

    print_summary(results)
    save_results(results, query)

    return results


def print_summary(results: dict):
    """打印所有论文的分析结果"""

    papers = results["papers"]
    print(f"\n主题: \"{results['query']}\"")
    print(f"共分析 {len(papers)} 篇论文\n")

    for i, paper in enumerate(papers, 1):
        analysis = paper.get("analysis", {})

        if "error" in analysis:
            print(f"[{i}] {paper['title']}")
            print(f"    ❌ 分析失败: {analysis['error']}\n")
            continue

        print(f"[{i}] {paper['title']} ({paper['year']})")
        print(f"    研究问题: {analysis.get('research_problem', 'N/A')}")
        print(f"    研究方法: {analysis.get('methodology', 'N/A')}")
        print(f"    主要贡献: {analysis.get('main_contribution', 'N/A')}")
        print(f"    关键结果: {analysis.get('key_results', 'N/A')}")
        print(f"    关键词:   {', '.join(analysis.get('keywords', []))}")
        print()


def save_results(results: dict, query: str):
    """
    保存结果到 JSON 文件

    === 为什么要保存？===
    1. 每次调 API 都花钱，保存后下次可以直接读取，不用重新分析
    2. 后面做"多文献归纳"时，直接读这个 JSON 就行
    3. 这就是你设计文档里的"缓存机制"
    """
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 用关键词作文件名（去掉特殊字符）
    safe_name = query.replace(" ", "_")[:50]
    file_path = os.path.join(output_dir, f"{safe_name}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {file_path}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 你可以修改这里的关键词来测试不同主题
    query = "transformer attention mechanism"

    # max_papers 建议 3-5 篇（省钱，测试够用）
    results = run_pipeline(query, max_papers=3)
