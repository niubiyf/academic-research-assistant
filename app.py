"""
大模型辅助的学术文献调研与观点归纳系统 — Web 界面

运行方式：
  cd v2_system
  streamlit run app.py

然后浏览器会自动打开 http://localhost:8501
"""

import os
import sys
import json
import time
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from modules.arxiv_searcher import search_papers, download_pdf
from modules.pdf_processor import process_pdf
from modules.paper_analyzer import PaperAnalyzer
from modules.multi_paper_summarizer import MultiPaperSummarizer


# ============================================================
# 页面配置（必须放在最前面）
# ============================================================
st.set_page_config(
    page_title="学术文献调研助手",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# API 配置
# ============================================================
# 从 .streamlit/secrets.toml 读取（部署到云端时安全，不会暴露 Key）
# 本地运行时也从同一个文件读取，保持一致
API_KEY = st.secrets["API_KEY"]
BASE_URL = st.secrets["BASE_URL"]
MODEL = st.secrets["MODEL"]
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")


# ============================================================
# 页面标题
# ============================================================
st.title("📚 学术文献调研与观点归纳系统")
st.caption("输入研究主题，自动搜索论文 → 智能分析 → 归纳观点 → 生成报告大纲")

st.divider()

# ============================================================
# 侧边栏：参数设置
# ============================================================
with st.sidebar:
    st.header("⚙️ 参数设置")

    max_papers = st.slider(
        "搜索论文数量",
        min_value=2, max_value=10, value=5,
        help="建议 3-5 篇，数量越多分析时间越长"
    )

    st.divider()

    st.subheader("📖 使用说明")
    st.markdown("""
    1. 在主页输入**英文**研究主题
    2. 点击「开始调研」
    3. 系统自动完成：
       - 🔍 arXiv 论文搜索
       - 📄 论文摘要提取
       - 🤖 AI 逐篇分析
       - 📊 多文献观点归纳
       - 📋 调研报告大纲生成
    """)

    st.divider()
    st.caption("毕业设计 · 大模型辅助的学术文献调研与观点归纳系统")


# ============================================================
# 主区域：输入 + 搜索
# ============================================================
query = st.text_input(
    "🔍 输入研究主题（英文关键词）",
    placeholder="例如：federated learning privacy protection",
)

start_btn = st.button("🚀 开始调研", type="primary", use_container_width=True)


# ============================================================
# 核心流程
# ============================================================
if start_btn and query:

    # ---- Step 1: 搜索 ----
    with st.status("正在调研中...", expanded=True) as status:

        st.write("🔍 **Step 1:** 搜索 arXiv 论文...")
        papers = search_papers(query, max_results=max_papers)

        if not papers:
            st.error("未找到相关论文，请尝试换个关键词")
            st.stop()

        st.write(f"✅ 找到 **{len(papers)}** 篇论文")

        # ---- Step 2: 下载 PDF ----
        st.write("📥 **Step 2:** 下载论文 PDF...")
        download_progress = st.progress(0)

        for i, paper in enumerate(papers):
            pdf_path = download_pdf(paper, save_dir=PAPERS_DIR)
            paper["local_pdf_path"] = pdf_path
            download_progress.progress((i + 1) / len(papers))
            if i < len(papers) - 1:
                time.sleep(3)

        st.write("✅ PDF 下载完成")

        # ---- Step 3: 准备摘要 ----
        st.write("📝 **Step 3:** 提取论文摘要...")
        for paper in papers:
            arxiv_abstract = paper.get("abstract", "")
            if len(arxiv_abstract) > 100:
                paper["final_abstract"] = arxiv_abstract
                paper["abstract_source"] = "arXiv API"
            else:
                pdf_path = paper.get("local_pdf_path")
                if pdf_path and os.path.exists(pdf_path):
                    pdf_data = process_pdf(pdf_path)
                    paper["final_abstract"] = pdf_data.get("abstract", arxiv_abstract)
                    paper["abstract_source"] = "PDF 提取"
                else:
                    paper["final_abstract"] = arxiv_abstract
                    paper["abstract_source"] = "arXiv API"

        st.write("✅ 摘要准备完成")

        # ---- Step 4: 逐篇分析 ----
        st.write("🤖 **Step 4:** AI 逐篇分析论文...")
        analyzer = PaperAnalyzer(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
        analyze_progress = st.progress(0)

        paper_results = []
        for i, paper in enumerate(papers):
            analysis = analyzer.analyze(
                title=paper["title"],
                abstract=paper["final_abstract"]
            )
            paper_results.append({
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper["title"],
                "authors": paper.get("authors", []),
                "year": paper.get("year", ""),
                "abstract_source": paper.get("abstract_source", ""),
                "abstract": paper["final_abstract"],
                "analysis": analysis,
            })
            analyze_progress.progress((i + 1) / len(papers))
            if i < len(papers) - 1:
                time.sleep(1)

        st.write("✅ 论文分析完成")

        # ---- Step 5: 多文献归纳 ----
        st.write("📊 **Step 5:** 多文献归纳（主流方法 + 优缺点 + 报告大纲）...")
        summarizer = MultiPaperSummarizer(
            api_key=API_KEY, base_url=BASE_URL, model=MODEL
        )
        summary = summarizer.summarize(paper_results)

        st.write("✅ 归纳完成")
        status.update(label="调研完成！", state="complete", expanded=False)

    # ============================================================
    # 结果展示
    # ============================================================

    st.divider()
    st.header("📊 调研结果")

    # ---- Tab 布局 ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 论文列表",
        "🔬 逐篇分析",
        "📊 观点归纳",
        "📄 调研报告大纲",
    ])

    # ---- Tab 1: 论文列表 ----
    with tab1:
        st.subheader(f"搜索到 {len(paper_results)} 篇论文")

        for i, paper in enumerate(paper_results, 1):
            with st.expander(f"**[{i}] {paper['title']}** ({paper['year']})"):
                authors = paper.get("authors", [])
                authors_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    authors_str += f" 等 {len(authors)} 人"

                st.markdown(f"**作者：** {authors_str}")
                st.markdown(f"**arXiv ID：** `{paper.get('arxiv_id', 'N/A')}`")
                st.markdown(f"**摘要来源：** {paper.get('abstract_source', 'N/A')}")
                st.markdown("**摘要：**")
                st.text(paper.get("abstract", "N/A")[:500])

    # ---- Tab 2: 逐篇分析 ----
    with tab2:
        st.subheader("AI 分析结果")

        for i, paper in enumerate(paper_results, 1):
            analysis = paper.get("analysis", {})

            if "error" in analysis:
                st.error(f"**[{i}] {paper['title']}** — 分析失败: {analysis['error']}")
                continue

            with st.expander(f"**[{i}] {paper['title']}** ({paper['year']})", expanded=(i == 1)):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**🎯 研究问题**")
                    st.info(analysis.get("research_problem", "N/A"))

                    st.markdown(f"**🔧 研究方法**")
                    st.info(analysis.get("methodology", "N/A"))

                    st.markdown(f"**💡 主要贡献**")
                    st.success(analysis.get("main_contribution", "N/A"))

                with col2:
                    st.markdown(f"**📈 关键结果**")
                    st.info(analysis.get("key_results", "N/A"))

                    st.markdown(f"**⚠️ 局限性**")
                    st.warning(analysis.get("limitations", "N/A"))

                    keywords = analysis.get("keywords", [])
                    if keywords:
                        st.markdown("**🏷️ 关键词**")
                        st.write(" · ".join(keywords))

    # ---- Tab 3: 观点归纳 ----
    with tab3:
        if "error" in summary:
            st.error(f"归纳失败: {summary['error']}")
        else:
            # 主流方法
            st.subheader("🔬 主流研究方法")
            methods = summary.get("main_methods", [])

            for i, method in enumerate(methods, 1):
                with st.expander(f"**方法 {i}：{method.get('name', 'Unknown')}**", expanded=True):
                    st.markdown(f"**核心思想：** {method.get('core_idea', 'N/A')}")

                    rep_papers = method.get("representative_papers", [])
                    if rep_papers:
                        st.markdown(f"**代表论文：** {'; '.join(rep_papers)}")

                    col_adv, col_dis = st.columns(2)
                    with col_adv:
                        st.markdown("**✅ 优点**")
                        for adv in method.get("advantages", []):
                            st.markdown(f"- {adv}")
                    with col_dis:
                        st.markdown("**❌ 缺点**")
                        for dis in method.get("disadvantages", []):
                            st.markdown(f"- {dis}")

            # 方法对比表
            comparison = summary.get("comparison_table", "")
            if comparison:
                st.subheader("📊 方法对比")
                st.markdown(comparison)

            # 研究趋势
            trends = summary.get("research_trends", "")
            if trends:
                st.subheader("📈 研究趋势")
                st.info(trends)

    # ---- Tab 4: 报告大纲 ----
    with tab4:
        outline = summary.get("report_outline", [])

        if outline:
            st.subheader(f"📄 「{query}」文献调研报告大纲")
            st.divider()

            for item in outline:
                if item.startswith("  "):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{item.strip()}")
                else:
                    st.markdown(f"**{item}**")

            st.divider()
            st.caption("以上大纲由 AI 基于论文分析结果自动生成")
        else:
            st.warning("未能生成报告大纲")

    # ---- 保存结果 ----
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    safe_name = query.replace(" ", "_")[:50]
    file_path = os.path.join(output_dir, f"{safe_name}.json")

    all_results = {
        "query": query,
        "papers": paper_results,
        "summary": summary,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    st.divider()
    st.success(f"📁 结果已保存到: `{file_path}`")

elif start_btn and not query:
    st.warning("请先输入研究主题")
