"""
PDF处理模块
功能：从学术论文PDF中提取摘要和关键信息
"""

import re
import pdfplumber

# PyMuPDF 作为备用方案，按需导入（部分版本有依赖问题）
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def extract_with_pdfplumber(pdf_path: str) -> str:
    """使用 pdfplumber 提取PDF文本"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:3]:
            # x_tolerance 控制空格判定的字符间距阈值，调大可以减少粘连
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    return text


def extract_with_pymupdf(pdf_path: str) -> str:
    """使用 PyMuPDF 提取PDF文本（备用方案）"""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF 不可用，请运行: pip install pymupdf-fonts")
    text = ""
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        if i >= 3:  # 只看前3页
            break
        text += page.get_text() + "\n"
    doc.close()
    return text


def extract_abstract(text: str) -> str:
    """
    从文本中提取摘要（Abstract）部分
    
    策略：用正则表达式找 Abstract 开头到 Keywords/Introduction 结尾
    """
    # 常见的摘要结束标志
    end_patterns = [
        r'Keywords?[:\s]',
        r'Index Terms?[:\s]',
        r'1[\.\s]+Introduction',
        r'I[\.\s]+Introduction',
        r'1[\.\s]+INTRODUCTION',
    ]
    end_pattern = '|'.join(end_patterns)

    # 尝试提取 Abstract 部分
    pattern = rf'Abstract[:\s\n]+(.*?)(?={end_pattern})'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        abstract = match.group(1).strip()
        abstract = clean_text(abstract)
        return abstract

    # 提取失败时，返回前2000字符作为降级方案
    return clean_text(text[:2000])


def fix_missing_spaces(text: str) -> str:
    """
    修复PDF提取时丢失的空格
    原理：检测小写字母紧跟大写字母（如 "modelThe" → "model The"）
    以及常见的粘连模式
    """
    # 小写字母紧跟大写字母，中间加空格（如 "modelThe" → "model The"）
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 字母紧跟数字 or 数字紧跟字母（如 "BLEU28" → "BLEU 28"）
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # 句号/逗号/分号后面紧跟字母（如 "results.We" → "results. We"）
    text = re.sub(r'([.,:;])([A-Za-z])', r'\1 \2', text)

    # 右括号后面紧跟字母（如 ")and" → ") and"）
    text = re.sub(r'\)([A-Za-z])', r') \1', text)

    # 字母紧跟左括号（如 "model(the" → "model (the"）
    text = re.sub(r'([a-z])\(', r'\1 (', text)

    return text


def clean_text(text: str) -> str:
    """
    清洗学术论文文本
    去除页眉页脚、修复换行、修复空格、统一空白字符
    """
    # 先修复丢失的空格
    text = fix_missing_spaces(text)

    # 修复连字符断行（如 "atten-\ntion" → "attention"）
    text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)

    # 合并段落内的换行
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 去除多余空白
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 去除常见页眉页脚关键词所在行
    lines = text.split('\n')
    filtered = []
    noise_keywords = ['Proceedings', 'Conference', 'IEEE', 'ACM', 'arXiv', 'preprint']
    for line in lines:
        if any(kw in line for kw in noise_keywords) and len(line) < 100:
            continue
        filtered.append(line)
    text = '\n'.join(filtered)

    return text.strip()


def extract_metadata(text: str) -> dict:
    """
    提取论文基础元数据（标题、作者等）
    注意：这是简单的启发式方法，不保证100%准确
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # 通常第一行是标题（较长且不是作者名格式）
    title = lines[0] if lines else "Unknown"

    # 尝试找年份（4位数字，1990-2030）
    year_match = re.search(r'\b(19[9]\d|20[0-3]\d)\b', text)
    year = year_match.group(1) if year_match else "Unknown"

    return {
        "title": title,
        "year": year,
    }


def process_pdf(pdf_path: str) -> dict:
    """
    主函数：处理单个PDF文件
    返回包含标题、年份、摘要的字典
    """
    print(f"正在处理：{pdf_path}")

    # 策略：优先 PyMuPDF（空格处理更好），失败用 pdfplumber
    raw_text = ""

    if PYMUPDF_AVAILABLE:
        try:
            raw_text = extract_with_pymupdf(pdf_path)
            if len(raw_text) < 100:
                raise ValueError("提取内容太少")
            print("  [OK] PyMuPDF 提取成功")
        except Exception as e:
            print(f"  [!] PyMuPDF 失败: {e}")
            raw_text = ""

    if len(raw_text) < 100:
        try:
            raw_text = extract_with_pdfplumber(pdf_path)
            if len(raw_text) < 100:
                raise ValueError("提取内容太少")
            print("  [OK] pdfplumber 提取成功")
        except Exception as e2:
            print(f"  [FAIL] 所有方法都失败了: {e2}")
            return {"error": str(e2)}

    # 提取摘要
    abstract = extract_abstract(raw_text)
    print(f"  [OK] 摘要提取成功（{len(abstract)} 字符）")

    # 提取元数据
    metadata = extract_metadata(raw_text)

    return {
        "title": metadata["title"],
        "year": metadata["year"],
        "abstract": abstract,
        "abstract_length": len(abstract),
    }


# ============================================================
# 测试代码
# ============================================================
def safe_print(text: str):
    """安全打印（Windows终端兼容，替换无法编码的字符）"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('gbk', errors='replace').decode('gbk'))


if __name__ == "__main__":
    import sys

    # 使用你已有的 Transformer 论文测试
    pdf_path = r"C:\Users\86158\Desktop\VSCode code\毕业设计\papers\1706.03762.pdf"

    result = process_pdf(pdf_path)

    print("\n" + "=" * 60)
    print("提取结果：")
    print("=" * 60)
    safe_print(f"标题：{result.get('title', 'N/A')}")
    safe_print(f"年份：{result.get('year', 'N/A')}")
    safe_print(f"摘要长度：{result.get('abstract_length', 0)} 字符")
    safe_print(f"\n摘要内容：\n{result.get('abstract', 'N/A')}")
