"""
论文分析模块
功能：调用大模型API，对单篇论文进行结构化分析
输出：JSON格式的分析结果
"""

import json
from openai import OpenAI


# ============================================================
# Prompt 模板（核心）
# ============================================================

# v1：基础版（结构化 + 强制JSON输出）
PROMPT_V1 = """你是一个学术论文分析专家。请分析以下论文摘要，严格按照JSON格式输出。

论文标题：{title}

论文摘要：
{abstract}

请提取以下信息（必须严格按JSON格式输出，不要输出任何JSON以外的内容）：
{{
  "research_problem": "该论文要解决什么核心问题？（50字以内）",
  "methodology": "使用了什么研究方法或技术？列出关键算法/模型名称（50字以内）",
  "main_contribution": "论文的主要创新点和贡献是什么？（80字以内）",
  "key_results": "主要实验结果或性能指标（如摘要中未提及则填'未提及'）（50字以内）",
  "limitations": "论文存在的局限性（如摘要中未提及则填'未提及'）（50字以内）",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}
"""

# v2：Few-shot版（提供示例，效果更好）
PROMPT_V2 = """你是一个学术论文分析专家。请参考以下示例，分析新论文。

【示例】
标题：Attention Is All You Need
摘要：The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...
输出：
{{
  "research_problem": "传统RNN/CNN序列模型计算复杂、难以并行化",
  "methodology": "Transformer架构，完全基于自注意力机制（Self-Attention）",
  "main_contribution": "提出不依赖RNN/CNN的Transformer模型，并行能力强，性能优越",
  "key_results": "WMT 2014英德翻译BLEU 28.4，超越已有最佳模型",
  "limitations": "未提及",
  "keywords": ["Transformer", "Self-Attention", "机器翻译", "序列模型"]
}}

【现在请分析以下论文】
标题：{title}
摘要：
{abstract}

请输出JSON格式的分析结果：
"""


class PaperAnalyzer:
    """论文分析器"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.prompt_version = "v2"  # 默认使用 Few-shot 版本

    def analyze(self, title: str, abstract: str) -> dict:
        """
        分析单篇论文
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            结构化分析结果（dict）
        """
        # 选择 Prompt 模板
        if self.prompt_version == "v1":
            prompt = PROMPT_V1.format(title=title, abstract=abstract)
        else:
            prompt = PROMPT_V2.format(title=title, abstract=abstract)

        print(f"  正在分析论文：{title[:50]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},  # 强制JSON输出
                temperature=0.3,   # 低温度 = 输出更稳定
                max_tokens=1000,
            )

            raw_result = response.choices[0].message.content
            result = json.loads(raw_result)

            # 添加元数据
            result["title"] = title
            result["prompt_version"] = self.prompt_version

            print(f"  [OK] 分析完成")
            return result

        except json.JSONDecodeError as e:
            print(f"  [!] JSON解析失败，尝试修复...")
            # 尝试从结果中提取JSON
            return self._try_fix_json(raw_result, title)

        except Exception as e:
            print(f"  [FAIL] API调用失败: {e}")
            return {"error": str(e), "title": title}

    def _try_fix_json(self, raw_text: str, title: str) -> dict:
        """尝试从非标准输出中提取JSON"""
        import re
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"error": "JSON解析失败", "raw": raw_text, "title": title}

    def format_result(self, result: dict) -> str:
        """将分析结果格式化为可读的Markdown"""
        if "error" in result:
            return f"分析失败: {result['error']}"

        md = f"""## {result.get('title', '未知标题')}

| 项目 | 内容 |
|------|------|
| **研究问题** | {result.get('research_problem', 'N/A')} |
| **研究方法** | {result.get('methodology', 'N/A')} |
| **主要贡献** | {result.get('main_contribution', 'N/A')} |
| **实验结果** | {result.get('key_results', 'N/A')} |
| **局限性** | {result.get('limitations', 'N/A')} |

**关键词**：{', '.join(result.get('keywords', []))}
"""
        return md


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 配置 API（使用 DeepSeek）
    API_KEY = "your-deepseek-api-key"  # ← 替换成你的 API Key
    
    # 用 Transformer 论文测试
    test_title = "Attention Is All You Need"
    test_abstract = """
    The dominant sequence transduction models are based on complex recurrent or 
    convolutional neural networks that include an encoder and a decoder. The best 
    performing models also connect the encoder and decoder through an attention 
    mechanism. We propose a new simple network architecture, the Transformer, 
    based solely on attention mechanisms, dispensing with recurrence and convolutions 
    entirely. Experiments on two machine translation tasks show these models to be 
    superior in quality while being more parallelizable and requiring significantly 
    less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German 
    translation task, improving over the existing best results, including ensembles, 
    by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model 
    establishes a new single-model state-of-the-art BLEU score of 41.0 after training 
    for 3.5 days on eight GPUs.
    """

    analyzer = PaperAnalyzer(api_key=API_KEY)

    print("=" * 60)
    print("测试 v1 基础版 Prompt")
    print("=" * 60)
    analyzer.prompt_version = "v1"
    result_v1 = analyzer.analyze(test_title, test_abstract)
    print(analyzer.format_result(result_v1))

    print("\n" + "=" * 60)
    print("测试 v2 Few-shot 版 Prompt")
    print("=" * 60)
    analyzer.prompt_version = "v2"
    result_v2 = analyzer.analyze(test_title, test_abstract)
    print(analyzer.format_result(result_v2))
