"""
arXiv 论文搜索模块
功能：根据用户输入的研究主题，从 arXiv 搜索相关论文

=== 原理说明 ===

arXiv 提供了免费的 HTTP API，任何人都可以用它搜索论文。
工作方式就像你在浏览器里搜索一样，只不过我们用代码发请求：

  你的代码  →  发 HTTP 请求  →  arXiv 服务器
  你的代码  ←  返回 XML 数据  ←  arXiv 服务器

API 地址：http://export.arxiv.org/api/query
参数说明：
  - search_query: 搜索关键词（支持 AND/OR 组合）
  - start: 从第几条开始（分页用）
  - max_results: 最多返回几条
  - sortBy: 排序方式（relevance=相关度, lastUpdatedDate=最新）

返回的是 Atom XML 格式（一种结构化数据），我们用 feedparser 库来解析。
"""

import re
import feedparser
import requests
import os
import time


# arXiv API 的基础地址（全球统一，免费，不需要注册）
ARXIV_API_URL = "http://export.arxiv.org/api/query"


def search_papers(query: str, max_results: int = 10,
                  sort_by: str = "relevance") -> list[dict]:
    """
    搜索 arXiv 论文

    === 这个函数做了什么？===
    1. 把你的中文/英文关键词拼成 arXiv 能理解的搜索 URL
    2. 发 HTTP GET 请求给 arXiv
    3. 解析返回的 XML，提取每篇论文的标题、摘要、作者等
    4. 整理成 Python 字典列表返回

    Args:
        query:       搜索关键词，如 "federated learning privacy"
        max_results: 最多返回多少篇（默认10，arXiv限制最大2000）
        sort_by:     排序方式 - "relevance"(相关度) 或 "lastUpdatedDate"(最新)

    Returns:
        论文列表，每篇是一个字典，包含 title, abstract, authors 等
    """

    # ---- 第1步：构造搜索 URL ----
    #
    # arXiv API 用 URL 参数传递搜索条件，就像你在百度搜索时 URL 会变成
    # baidu.com/s?wd=xxx 一样，arXiv 的格式是：
    # http://export.arxiv.org/api/query?search_query=all:xxx&max_results=10
    #
    # "all:" 表示在所有字段（标题+摘要+全文）中搜索
    # 空格要替换成 "+"（URL编码规则）

    search_query = f"all:{query.replace(' ', '+')}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    print(f"正在搜索 arXiv: \"{query}\" (最多 {max_results} 篇)...")

    # ---- 第2步：发送 HTTP 请求 ----
    #
    # requests.get() 就是在代码里"打开一个网页"
    # 它会返回网页的内容（这里是 XML 格式的论文数据）
    #
    # timeout=30 表示如果30秒没响应就放弃（防止一直卡住）

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()  # 如果HTTP状态码不是200，抛出异常
    except requests.RequestException as e:
        print(f"  [FAIL] 网络请求失败: {e}")
        return []

    # ---- 第3步：解析返回的 XML 数据 ----
    #
    # arXiv 返回的是 Atom XML 格式（一种类似 HTML 的标记语言）
    # 直接手动解析 XML 很麻烦，所以用 feedparser 库自动解析
    # feedparser 会把 XML 变成 Python 对象，方便我们读取

    feed = feedparser.parse(response.text)

    if not feed.entries:
        print("  未找到相关论文")
        return []

    # ---- 第4步：提取每篇论文的关键信息 ----
    #
    # feed.entries 是一个列表，每个元素代表一篇论文
    # 我们从中提取需要的字段，整理成统一格式

    papers = []

    for entry in feed.entries:
        paper = _parse_entry(entry)
        papers.append(paper)

    print(f"  [OK] 找到 {len(papers)} 篇论文")
    return papers


def _parse_entry(entry) -> dict:
    """
    解析单条 arXiv 搜索结果

    === entry 里有什么？===
    feedparser 解析后，每篇论文（entry）是一个类字典对象，包含：
    - entry.title        → 论文标题
    - entry.summary      → 论文摘要
    - entry.authors      → 作者列表
    - entry.published    → 发表日期（如 "2023-06-15T17:59:01Z"）
    - entry.links        → 链接列表（包含 PDF 下载链接）
    - entry.id           → arXiv 唯一标识（如 "http://arxiv.org/abs/2306.09299v1"）
    """

    # 提取标题（去掉多余的换行和空格）
    title = entry.get("title", "Unknown")
    title = " ".join(title.split())  # "hello\n  world" → "hello world"

    # 提取摘要
    abstract = entry.get("summary", "")
    abstract = " ".join(abstract.split())

    # 提取作者列表
    # entry.authors 格式：[{"name": "John"}, {"name": "Jane"}]
    authors = [a.get("name", "") for a in entry.get("authors", [])]

    # 提取发表年份
    # entry.published 格式："2023-06-15T17:59:01Z"
    # 我们只要前4个字符（年份）
    published = entry.get("published", "")
    year = published[:4] if len(published) >= 4 else "Unknown"

    # 提取 arXiv ID
    # entry.id 格式："http://arxiv.org/abs/2306.09299v1"
    # 我们要的是最后的 "2306.09299v1" → 去掉版本号 → "2306.09299"
    arxiv_url = entry.get("id", "")
    arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""
    arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)  # 去掉 v1, v2 等

    # 构造 PDF 下载链接
    # arXiv 的 PDF 链接规则：https://arxiv.org/pdf/{id}.pdf
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf" if arxiv_id_clean else ""

    # 提取分类标签（如 cs.AI, cs.CL）
    tags = [t.get("term", "") for t in entry.get("tags", [])]

    return {
        "arxiv_id": arxiv_id_clean,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "published": published[:10],  # 只保留日期部分 "2023-06-15"
        "pdf_url": pdf_url,
        "arxiv_url": arxiv_url,
        "tags": tags,
    }


def download_pdf(paper: dict, save_dir: str = "papers") -> str | None:
    """
    下载单篇论文的 PDF

    === 原理 ===
    和你在浏览器点"下载"一样，代码用 requests.get() 获取 PDF 文件的二进制内容，
    然后用 open(..., "wb") 写入本地文件。

    "wb" = write binary（写入二进制），因为 PDF 不是文本，是二进制文件。

    stream=True 表示"流式下载"——不是一次性把整个文件读到内存，
    而是一小块一小块地读，这样下载大文件时不会把内存撑爆。

    Args:
        paper:    论文字典（必须包含 pdf_url 和 arxiv_id）
        save_dir: 保存目录

    Returns:
        保存的文件路径，失败返回 None
    """
    pdf_url = paper.get("pdf_url", "")
    arxiv_id = paper.get("arxiv_id", "unknown")

    if not pdf_url:
        print(f"  [!] 无 PDF 链接: {paper.get('title', '')}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{arxiv_id}.pdf")

    if os.path.exists(file_path):
        print(f"  [OK] 已存在，跳过: {arxiv_id}.pdf")
        return file_path

    print(f"  正在下载: {arxiv_id}.pdf ...")

    try:
        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  [OK] 下载成功: {file_path}")
        return file_path

    except Exception as e:
        print(f"  [FAIL] 下载失败: {e}")
        return None


def search_and_download(query: str, max_results: int = 5,
                        save_dir: str = "papers") -> list[dict]:
    """
    搜索 + 下载一步到位

    这是一个"便捷函数"，把 search_papers() 和 download_pdf() 组合起来。
    搜索完自动下载所有 PDF，并把本地路径写入每篇论文的字典里。

    === 为什么每篇下载之间要 sleep(3)？===
    arXiv API 有速率限制（rate limit）：不能太频繁地发请求。
    官方建议每次请求间隔至少 3 秒，否则可能被临时封禁 IP。
    这就像排队买东西，你不能插队，得一个一个来。

    Args:
        query:       搜索关键词
        max_results: 最多几篇
        save_dir:    PDF 保存目录

    Returns:
        带有 local_pdf_path 的论文列表
    """
    papers = search_papers(query, max_results=max_results)

    if not papers:
        return []

    print(f"\n开始下载 {len(papers)} 篇论文的 PDF...")

    for i, paper in enumerate(papers):
        pdf_path = download_pdf(paper, save_dir=save_dir)
        paper["local_pdf_path"] = pdf_path

        if i < len(papers) - 1:
            print(f"  等待 3 秒（arXiv 速率限制）...")
            time.sleep(3)

    downloaded = sum(1 for p in papers if p.get("local_pdf_path"))
    print(f"\n[OK] 下载完成: {downloaded}/{len(papers)} 篇")

    return papers


def display_results(papers: list[dict]):
    """格式化打印搜索结果"""
    if not papers:
        print("没有搜索结果")
        return

    print(f"\n{'='*70}")
    print(f"搜索到 {len(papers)} 篇论文")
    print(f"{'='*70}")

    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += f" 等 {len(paper['authors'])} 人"

        print(f"\n[{i}] {paper['title']}")
        print(f"    作者: {authors_str}")
        print(f"    年份: {paper['year']}  |  ID: {paper['arxiv_id']}")
        print(f"    分类: {', '.join(paper['tags'][:3])}")
        print(f"    摘要: {paper['abstract'][:150]}...")
        print(f"    PDF:  {paper['pdf_url']}")

    print(f"\n{'='*70}")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # ---- 测试1：只搜索，不下载 ----
    print("=" * 70)
    print("测试：搜索 arXiv 论文")
    print("=" * 70)

    papers = search_papers("federated learning privacy", max_results=5)
    display_results(papers)

    # ---- 测试2：搜索 + 下载第一篇 ----
    if papers:
        print("\n\n测试：下载第一篇论文")
        print("-" * 70)
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "papers"
        )
        pdf_path = download_pdf(papers[0], save_dir=save_dir)
        if pdf_path:
            print(f"\n下载到: {pdf_path}")
