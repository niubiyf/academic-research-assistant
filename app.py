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
from openai import OpenAI

sys.path.append(os.path.dirname(__file__))

from modules.arxiv_searcher import search_papers, download_pdf
from modules.pdf_processor import process_pdf
from modules.paper_analyzer import PaperAnalyzer
from modules.multi_paper_summarizer import MultiPaperSummarizer


def check_api_health(api_key: str, base_url: str, model: str) -> dict:
    """对 LLM 接口做一次最小调用，返回鉴权与可用性结果。"""
    if not api_key:
        return {"ok": False, "error": "未配置 API_KEY"}
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4,
            temperature=0,
        )
        usage = response.usage
        return {
            "ok": True,
            "model": model,
            "reply": response.choices[0].message.content,
            "tokens": getattr(usage, "total_tokens", None),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


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

col_check, col_start = st.columns([1, 3])
with col_check:
    check_btn = st.button("🩺 API 预检", use_container_width=True)
with col_start:
    start_btn = st.button("🚀 开始调研", type="primary", use_container_width=True)


# ============================================================
# API 预检：在跑全流程前先确认大模型接口可用
# ============================================================
if check_btn:
    with st.spinner("正在检查 API 配置与可用性..."):
        result = check_api_health(API_KEY, BASE_URL, MODEL)

    if result["ok"]:
        st.success(
            f"API 可用 ✅  模型：{result['model']}  "
            f"测试 token：{result.get('tokens', 'N/A')}"
        )
    else:
        st.error(
            "API 调用失败 ❌\n\n"
            f"错误信息：{result['error']}\n\n"
            "请检查 Streamlit Cloud → Manage app → Secrets 中的 "
            "`API_KEY` / `BASE_URL` / `MODEL` 是否正确，并重启应用。"
        )


# ============================================================
# 核心流程
# ============================================================
if start_btn and query:

    # ---- Step 0: API 预检 ----
    with st.spinner("Step 0: API 预检中..."):
        precheck = check_api_health(API_KEY, BASE_URL, MODEL)

    if not precheck["ok"]:
        st.error(
            "API 预检失败 ❌，已中止后续流程。\n\n"
            f"错误信息：{precheck['error']}\n\n"
            "请检查 Streamlit Cloud → Manage app → Secrets 中的 "
            "`API_KEY` / `BASE_URL` / `MODEL` 是否正确，并重启应用。"
        )
        st.stop()

    st.success(f"API 可用 ✅  模型：{precheck['model']}")

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
        summary = summarizer.summarize(paper_results, topic=query)

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
            if not methods:
                st.info("未识别到主流方法，已跳过该区域展示。")
            else:
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
        outline_raw = summary.get("report_outline", [])
        if isinstance(outline_raw, str):
            outline = [line.strip(" -\t") for line in outline_raw.splitlines() if line.strip()]
        elif isinstance(outline_raw, list):
            outline = []
            for item in outline_raw:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        outline.append(s)
                elif isinstance(item, dict):
                    title = str(item.get("title", "")).strip()
                    if title:
                        outline.append(title)
        else:
            outline = []

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

        # ---- 大纲内容（新增）----
        sections = summary.get("report_sections", [])
        if isinstance(sections, list) and sections:
            st.subheader("🧾 大纲章节内容（自动综合）")
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                title = sec.get("title", "未命名章节")
                with st.expander(f"{title}", expanded=True):
                    summary_text = str(sec.get("summary", "")).strip()
                    if summary_text:
                        st.write(summary_text)

                    bullets = sec.get("bullets", [])
                    if isinstance(bullets, list) and bullets:
                        st.markdown("**要点：**")
                        for b in bullets:
                            st.markdown(f"- {b}")

                    refs = sec.get("references", [])
                    if isinstance(refs, list) and refs:
                        st.markdown("**参考论文：**")
                        for r in refs:
                            st.markdown(f"- {r}")

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
    json_str = json.dumps(all_results, ensure_ascii=False, indent=2)
    st.download_button(
        label="⬇️ 下载本次结果(JSON)",
        data=json_str,
        file_name=f"{safe_name}.json",
        mime="application/json",
    )

    # 云端 outputs 不会自动同步到本地，提供下载按钮便于保存到本机
    json_str = json.dumps(all_results, ensure_ascii=False, indent=2)
    st.download_button(
        label="⬇️ 下载本次结果(JSON)",
        data=json_str,
        file_name=f"{safe_name}.json",
        mime="application/json",
    )

elif start_btn and not query:
    st.warning("请先输入研究主题")
