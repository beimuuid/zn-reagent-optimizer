import requests
import json
import base64
import mimetypes
import os
import time

# API 配置
BASE_URL = "http://35.220.164.252:3888"
API_KEY = "sk-IDOBL44LKUsHf4bhY5z0NMiLRjtSBZuYlNG9yzHUcelBIr7S"
MODEL_NAME = "gemini-2.5-pro"

# 构建 API 端点和请求头
API_URL = f"{BASE_URL}/v1/chat/completions"
HEADERS = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
}


def to_data_url(image_path: str) -> str:
    """将本地图片文件转换为 Base64 编码的 Data URL"""
    if not os.path.exists(image_path):
        print(f"警告: 找不到图片文件 {image_path}")
        return ""
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_gemini_api(prompt: str, image_path: str = None, system_prompt: str = None, 
                    model_name: str = None, temperature: float = None, 
                    base_url: str = None, api_key: str = None,
                    thinking_mode: bool = True, max_retries: int = 5, retry_delay: int = 10):
    """
    调用 Gemini 2.5 Pro API（带重试机制）
    
    参数:
        prompt: 用户输入的文本提示
        image_path: 可选的图片路径（支持多模态）
        system_prompt: 可选的系统提示词
        thinking_mode: 是否启用思考模式
        max_retries: 最大重试次数（默认3次）
        retry_delay: 重试延迟（秒，默认5秒）
    
    返回:
        API 响应的完整内容
    """
    # 优先使用传入的参数，否则使用全局默认值
    final_model_name = model_name or MODEL_NAME
    
    # 智能拼接 URL，移除 base_url 末尾的斜杠和 /v1/
    temp_base_url = (base_url or BASE_URL).rstrip('/').replace('/v1', '')
    final_api_url = f"{temp_base_url}/v1/chat/completions"

    final_headers = HEADERS.copy()
    if api_key:
        final_headers['Authorization'] = f'Bearer {api_key}'

    # 构建消息列表
    messages = []
    
    # 添加系统提示词
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # 构建用户消息内容
    user_content = []
    
    # 添加文本内容
    if prompt:
        user_content.append({"type": "text", "text": prompt})
    
    # 添加图片内容（如果提供）
    if image_path:
        img_data_url = to_data_url(image_path)
        if img_data_url:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": img_data_url, "detail": "auto"}
            })
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    # 构建请求 payload
    payload = {
        "model": final_model_name,
        "messages": messages,
        "thinking_mode": thinking_mode
    }
    if temperature is not None:
        payload["temperature"] = temperature
    
    # 带重试的 API 请求
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(final_api_url, headers=final_headers, json=payload, timeout=600)
            # 检查状态码
            if response.status_code == 200:
                data = response.json()
                # 提取响应内容
                content = data['choices'][0]['message']['content']
                return content
            elif response.status_code >= 500:
                # 服务器错误，可以重试
                error_msg = f"服务器错误 ({response.status_code})"
                if attempt < max_retries:
                    print(f"⚠️ {error_msg}，{retry_delay}秒后重试 ({attempt}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ {error_msg}，已达到最大重试次数")
                    try:
                        error_data = response.json()
                        print(f"错误详情: {json.dumps(error_data, ensure_ascii=False)}")
                    except:
                        print(f"响应内容: {response.text[:500]}")
                    return None
            else:
                # 其他错误（4xx等），不重试
                response.raise_for_status()
                
        except requests.exceptions.Timeout:
            # 打印请求超时信息
            if attempt < max_retries:
                print(f"⚠️ 请求超时，{retry_delay}秒后重试 ({attempt}/{max_retries})...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"❌ 请求超时，已达到最大重试次数")
                return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"⚠️ API 请求错误: {e}，{retry_delay}秒后重试 ({attempt}/{max_retries})...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"❌ API 请求错误: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"响应内容: {e.response.text[:500]}")
                return None
        except Exception as e:
            print(f"❌ 处理响应时出错: {e}")
            return None
    
    return None


def main():
    """示例使用"""
    # 示例 1: 纯文本对话
    print("=" * 50)
    print("示例 1: 纯文本对话")
    print("=" * 50)
    response = call_gemini_api(
        prompt="请简单介绍一下聚合物的基本概念。",
        system_prompt="你是一个聚合物科学专家。"
    )
    if response:
        print("响应内容:")
        print(response)
        print()
    
if __name__ == "__main__":
    main()

