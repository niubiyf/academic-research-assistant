"""
多文献归纳模块
功能：把多篇论文的分析结果归纳总结，输出主流方法、优缺点、调研报告大纲

=== 核心原理：Map-Reduce ===

这是你设计文档里的核心策略，灵感来自大数据处理框架。

Map 阶段（已完成）：逐篇分析每篇论文 → 得到每篇的方法、贡献
    论文A → {方法: "差分隐私", 贡献: "..."}
    论文B → {方法: "安全多方计算", 贡献: "..."}
    论文C → {方法: "差分隐私", 贡献: "..."}

Reduce 阶段（本模块）：把所有分析结果汇总，让大模型归纳
    [论文A结果, 论文B结果, 论文C结果]
        ↓ 大模型
    "主流方法有3个：1.差分隐私 2.安全多方计算 3.同态加密"

=== 为什么不把所有论文原文丢给大模型？===

因为 token 限制。如果直接把 5 篇论文的完整摘要都塞进去：
- 5 篇 × 1000 字符 = 5000 字符 + Prompt 指令 = 可能超限
- 而且原文里有很多重复和冗余信息

Map-Reduce 的优势：
- Map 阶段把每篇论文压缩成几十字的结构化信息
- Reduce 阶段只需要处理压缩后的信息，token 消耗大幅降低
- 5篇论文的 JSON 分析结果可能只有 1000 token，完全在限制内
"""

import json
from openai import OpenAI


# ============================================================
# Prompt 模板
# ============================================================

# 归纳 Prompt：让大模型从多篇论文的分析中提炼共性
SUMMARIZE_PROMPT = """你是一个学术调研专家。我给你 {num_papers} 篇论文的分析结果，请进行归纳总结。

## 论文分析列表

{papers_json}

## 请完成以下三个任务

### 任务1：识别主流研究方法（3-5个）
从这些论文中，归纳出该领域的主流研究方法/技术路线。
每个方法包括：
- 方法名称
- 核心思想（一句话）
- 代表论文（从上面的列表中选）
- 优点
- 缺点

### 任务2：生成方法对比表
用 Markdown 表格对比各方法：
| 方法 | 核心思想 | 优点 | 缺点 | 代表论文 |

### 任务3：生成调研报告大纲
基于以上分析，生成一份文献调研报告的大纲结构。

## 输出要求
请严格按以下 JSON 格式输出：
{{
  "main_methods": [
    {{
      "name": "方法名称",
      "core_idea": "核心思想（30字内）",
      "representative_papers": ["论文标题1"],
      "advantages": ["优点1", "优点2"],
      "disadvantages": ["缺点1", "缺点2"]
    }}
  ],
  "comparison_table": "Markdown 格式的对比表格",
  "report_outline": [
    "一、研究背景与意义",
    "二、主流研究方法",
    "  2.1 方法一：xxx",
    "  2.2 方法二：xxx",
    "三、方法对比分析",
    "四、研究趋势与展望",
    "五、参考文献"
  ],
  "research_trends": "对该领域研究趋势的简要分析（100字内）"
}}
"""


class MultiPaperSummarizer:
    """多文献归纳器"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def summarize(self, paper_analyses: list[dict]) -> dict:
        """
        归纳多篇论文的分析结果

        === 这个函数做了什么？===
        1. 把每篇论文的分析结果精简成"方法+贡献"的摘要
        2. 把所有摘要打包成一个 Prompt
        3. 让大模型一次性归纳出主流方法、优缺点、报告大纲

        Args:
            paper_analyses: 论文分析结果列表，每个元素是 pipeline.py 输出的字典

        Returns:
            归纳结果字典
        """

        if not paper_analyses:
            return {"error": "没有论文分析结果可归纳"}

        # ---- 第1步：精简每篇论文的信息 ----
        #
        # 不需要把所有字段都发给大模型，只保留归纳需要的核心信息
        # 这样可以节省 token，也让大模型更聚焦
        simplified = []
        for p in paper_analyses:
            analysis = p.get("analysis", {})
            if "error" in analysis:
                continue

            simplified.append({
                "title": p.get("title", "Unknown"),
                "year": p.get("year", "Unknown"),
                "research_problem": analysis.get("research_problem", ""),
                "methodology": analysis.get("methodology", ""),
                "main_contribution": analysis.get("main_contribution", ""),
                "key_results": analysis.get("key_results", ""),
                "keywords": analysis.get("keywords", []),
            })

        if not simplified:
            return {"error": "所有论文分析都失败了，无法归纳"}

        # ---- 第2步：构造 Prompt ----
        papers_json = json.dumps(simplified, ensure_ascii=False, indent=2)

        prompt = SUMMARIZE_PROMPT.format(
            num_papers=len(simplified),
            papers_json=papers_json
        )

        print(f"正在归纳 {len(simplified)} 篇论文的研究观点...")

        # ---- 第3步：调用大模型 ----
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=3000,
            )

            raw_result = response.choices[0].message.content
            result = json.loads(raw_result)

            print("  [OK] 归纳完成")
            return result

        except json.JSONDecodeError:
            print("  [!] JSON 解析失败，尝试修复...")
            return self._try_fix_json(raw_result)

        except Exception as e:
            print(f"  [FAIL] 归纳失败: {e}")
            return {"error": str(e)}

    def _try_fix_json(self, raw_text: str) -> dict:
        """尝试从非标准输出中提取 JSON"""
        import re
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"error": "JSON 解析失败", "raw": raw_text}

    def format_result(self, result: dict) -> str:
        """
        把归纳结果格式化成可读的 Markdown

        这个方法生成的内容可以直接展示在 Streamlit 界面上，
        也可以粘贴到你的论文里当示例
        """
        if "error" in result:
            return f"归纳失败: {result['error']}"

        md = ""

        # ---- 主流方法 ----
        md += "## 主流研究方法\n\n"
        methods = result.get("main_methods", [])
        for i, method in enumerate(methods, 1):
            md += f"### 方法 {i}：{method.get('name', 'Unknown')}\n"
            md += f"- **核心思想**：{method.get('core_idea', 'N/A')}\n"

            papers = method.get("representative_papers", [])
            if papers:
                md += f"- **代表论文**：{'; '.join(papers)}\n"

            advantages = method.get("advantages", [])
            if advantages:
                md += f"- **优点**：\n"
                for adv in advantages:
                    md += f"  - {adv}\n"

            disadvantages = method.get("disadvantages", [])
            if disadvantages:
                md += f"- **缺点**：\n"
                for dis in disadvantages:
                    md += f"  - {dis}\n"

            md += "\n"

        # ---- 方法对比表 ----
        comparison = result.get("comparison_table", "")
        if comparison:
            md += "## 方法对比\n\n"
            md += comparison + "\n\n"

        # ---- 研究趋势 ----
        trends = result.get("research_trends", "")
        if trends:
            md += "## 研究趋势\n\n"
            md += trends + "\n\n"

        # ---- 调研报告大纲 ----
        outline = result.get("report_outline", [])
        if outline:
            md += "## 调研报告大纲\n\n"
            for item in outline:
                md += f"{item}\n"
            md += "\n"

        return md


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import os

    # 读取 pipeline 保存的 JSON 结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    if not json_files:
        print("没有找到分析结果文件，请先运行 pipeline.py")
        exit()

    # 使用最新的结果文件
    latest_file = os.path.join(output_dir, json_files[-1])
    print(f"读取文件: {latest_file}")

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 配置 API（和 pipeline.py 一样）
    # 本地测试时在这里填你的 Key，部署时由 app.py 通过 st.secrets 传入
    API_KEY = "your-api-key"
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    MODEL = "glm-4-flash"

    summarizer = MultiPaperSummarizer(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL
    )

    result = summarizer.summarize(data["papers"])

    print("\n" + "=" * 60)
    print("归纳结果")
    print("=" * 60)
    print(summarizer.format_result(result))

    print("\n" + "=" * 60)
    print("原始 JSON")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
