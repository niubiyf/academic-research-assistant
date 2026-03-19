"""
安全加载 API 配置：优先从环境变量或 .streamlit/secrets.toml 读取，避免硬编码
"""
import os
import re


def load_api_config():
    """
    加载 API 配置，优先级：环境变量 > .streamlit/secrets.toml 文件
    
    Returns:
        dict: {"api_key": str, "base_url": str, "model": str}
    """
    api_key = os.environ.get("API_KEY") or os.environ.get("ZHIPU_API_KEY")
    base_url = os.environ.get("BASE_URL")
    model = os.environ.get("MODEL")

    if not api_key:
        secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as f:
                content = f.read()
            api_key = _parse_toml_value(content, "API_KEY")
            base_url = base_url or _parse_toml_value(content, "BASE_URL")
            model = model or _parse_toml_value(content, "MODEL")

    return {
        "api_key": api_key or "",
        "base_url": base_url or "https://open.bigmodel.cn/api/paas/v4",
        "model": model or "glm-4-flash",
    }


def _parse_toml_value(content: str, key: str) -> str | None:
    """从 TOML 内容中解析简单键值（仅支持 key = "value" 格式）"""
    pattern = rf'{key}\s*=\s*"([^"]*)"'
    match = re.search(pattern, content)
    return match.group(1) if match else None
