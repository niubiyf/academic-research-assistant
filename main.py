"""
主程序：将 PDF处理 + 论文分析 串联起来
运行方式：python main.py
"""

import sys
import os

# 把 modules 目录加入路径
sys.path.append(os.path.dirname(__file__))

from modules.pdf_processor import process_pdf
from modules.paper_analyzer import PaperAnalyzer

# ============================================================
# 配置区域
# ============================================================

# DeepSeek API（推荐，3元/百万token）
DEEPSEEK_API_KEY = "your-deepseek-api-key"   # ← 替换成你的 Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 或者用智谱 GLM（你的 demo 里已有 Key）
ZHIPU_API_KEY = "b021644ce4f3460fb5110696128f586f.TXB1AXj4XuLU9wjc"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_MODEL = "glm-4-flash"  # 用 flash 版本，更快更省钱

# ============================================================
# 主流程
# ============================================================

def analyze_paper(pdf_path: str):
    """
    完整流程：PDF → 摘要提取 → 大模型分析 → 输出结果
    """
    print("=" * 60)
    print("Step 1: 提取 PDF 内容")
    print("=" * 60)

    # 第一步：PDF 处理
    paper_data = process_pdf(pdf_path)

    if "error" in paper_data:
        print(f"❌ PDF处理失败：{paper_data['error']}")
        return

    print(f"\n标题：{paper_data['title']}")
    print(f"年份：{paper_data['year']}")
    print(f"摘要长度：{paper_data['abstract_length']} 字符")
    print(f"摘要预览：{paper_data['abstract'][:200]}...")

    print("\n" + "=" * 60)
    print("Step 2: 大模型分析")
    print("=" * 60)

    # 第二步：大模型分析
    # 使用智谱 GLM（你已有 Key，直接可用）
    analyzer = PaperAnalyzer(
        api_key=ZHIPU_API_KEY,
        base_url=ZHIPU_BASE_URL,
        model=ZHIPU_MODEL
    )

    result = analyzer.analyze(
        title=paper_data["title"],
        abstract=paper_data["abstract"]
    )

    print("\n" + "=" * 60)
    print("分析结果（Markdown格式）")
    print("=" * 60)
    print(analyzer.format_result(result))

    print("\n" + "=" * 60)
    print("分析结果（JSON格式，用于后续处理）")
    print("=" * 60)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


if __name__ == "__main__":
    # 使用你已有的 Transformer 论文测试
    pdf_path = r"C:\Users\86158\Desktop\VSCode code\毕业设计\papers\1706.03762.pdf"

    result = analyze_paper(pdf_path)
