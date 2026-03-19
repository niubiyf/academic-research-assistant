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
# 配置区域（从环境变量或 .streamlit/secrets.toml 读取，不硬编码密钥）
# ============================================================

from config_loader import load_api_config

_CONFIG = load_api_config()
API_KEY = _CONFIG["api_key"]
BASE_URL = _CONFIG["base_url"]
MODEL = _CONFIG["model"]

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
    if not API_KEY or API_KEY.startswith("your-"):
        print("❌ 请配置 API 密钥：在 .streamlit/secrets.toml 中设置 API_KEY，或设置环境变量 API_KEY")
        return None
    analyzer = PaperAnalyzer(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL
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
