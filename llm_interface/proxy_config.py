# llm_market_sim/llm_interface/proxy_config.py
import os
import logging

def set_proxy():
    """设置系统代理"""
    # 直接设置代理环境变量
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
    
    logging.debug(f"已设置系统代理: https_proxy=http://127.0.0.1:7890, http_proxy=http://127.0.0.1:7890, all_proxy=socks5://127.0.0.1:7890")
    # 注意：google-generativeai库通常会自动检测https_proxy环境变量

def unset_proxy():
    """取消代理设置"""
    # 清除代理环境变量
    if 'https_proxy' in os.environ:
        del os.environ['https_proxy']
    if 'http_proxy' in os.environ:
        del os.environ['http_proxy']
    if 'all_proxy' in os.environ:
        del os.environ['all_proxy']
        
    logging.debug("LLM请求完成，代理已释放。")

# 可以在这里添加更复杂的代理管理逻辑，例如强制设置requests库的代理等
# 但对于google-generativeai，环境变量通常足够