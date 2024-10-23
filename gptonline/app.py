import os
import openai
import requests
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient  # 确保 TavilyClient 已正确安装和配置
from bs4 import BeautifulSoup
from openai.error import RateLimitError, OpenAIError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载环境变量
load_dotenv()

# 从环境变量中获取API密钥
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# 检查API密钥是否存在
if not OPENAI_API_KEY:
    raise ValueError("未找到 OPENAI_API_KEY，请在环境变量中设置。")
if not TAVILY_API_KEY:
    raise ValueError("未找到 TAVILY_API_KEY，请在环境变量中设置。")

# 配置OpenAI API密钥
openai.api_key = OPENAI_API_KEY

# 初始化TavilyClient
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def crawl_web(query, num_results=5):
    """
    智能体1：使用TavilyClient的API来获取相关网页链接。
    """
    try:
        logging.info("智能体1：开始爬取相关网页链接。")
        data = tavily_client.search(query)

        # 检查响应中是否包含 'results'
        if 'results' not in data:
            raise Exception("Tavily搜索响应中缺少 'results' 键。")

        # 提取搜索结果中的链接
        links = [result.get('url') for result in data.get('results', []) if 'url' in result][:num_results]
        logging.info(f"智能体1：爬取到 {len(links)} 个网页链接。")
        return links
    except Exception as e:
        logging.error(f"智能体1：网页爬取时出错：{e}")
        return []


def fetch_page_content(url):
    """
    智能体2：获取单个网页的内容并提取主要文本。
    """
    try:
        logging.info(f"智能体2：开始获取 {url} 的内容。")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                          "AppleWebKit/537.36 (KHTML, like Gecko) " \
                          "Chrome/58.0.3029.110 Safari/537.3"
        }
        page_response = requests.get(url, headers=headers, timeout=10)
        if page_response.status_code == 200:
            soup = BeautifulSoup(page_response.text, 'html.parser')
            # 针对特定网站的内容提取（根据需要调整）
            if "zhihu.com" in url:
                # 示例：提取知乎文章的主要内容
                main_content = soup.find('div', {'class': 'RichContent-inner'})
                if main_content:
                    paragraphs = main_content.find_all('p')
                    main_text = '\n'.join([para.get_text() for para in paragraphs])
                else:
                    main_text = '\n'.join([para.get_text() for para in soup.find_all('p')])
            else:
                # 通用提取
                paragraphs = soup.find_all('p')
                main_text = '\n'.join([para.get_text() for para in paragraphs])

            # 可选：截断内容以限制长度
            max_length = 5000  # 根据需要调整
            if len(main_text) > max_length:
                main_text = main_text[:max_length]
            logging.info(f"智能体2：成功获取并处理了 {url} 的内容。")
            return main_text
        else:
            logging.warning(f"智能体2：无法获取 {url} 的内容，状态码：{page_response.status_code}")
            return ""
    except Exception as e:
        logging.error(f"智能体2：获取 {url} 内容时出错：{e}")
        return ""


def call_openai_chat_completion(messages, max_retries=5):
    """
    调用OpenAI的ChatCompletion API，并处理速率限制。
    """
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,  # 根据需要调整
                temperature=0.5,
            )
            return response.choices[0].message['content'].strip()
        except RateLimitError as e:
            wait_time = 2 ** retries
            logging.warning(f"速率限制错误：{e}. 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
            retries += 1
        except OpenAIError as e:
            logging.error(f"OpenAI API 错误：{e}")
            return "抱歉，我在处理您的请求时遇到了问题。"
    logging.error("达到最大重试次数，无法完成请求。")
    return "抱歉，我无法完成您的请求。"


def summarize_individual_content(content):
    """
    智能体2：使用OpenAI的ChatCompletion模型对单个网页内容进行摘要。
    """
    messages = [
        {"role": "system", "content": "你是一个帮助用户总结网页内容的助手。请用简洁的中文总结以下内容，不超过150字。"},
        {"role": "user", "content": f"请阅读以下内容并提供一个简洁的摘要（不超过150字）：\n\n{content}"}
    ]
    summary = call_openai_chat_completion(messages)
    if summary:
        logging.info(f"智能体2：摘要结果：{summary}")
    else:
        logging.warning("智能体2：无法生成摘要。")
    return summary if summary else "抱歉，我无法摘要所提供的内容。"


def summarize_all_contents(summaries):
    """
    智能体3：对所有单独的摘要进行综合总结。
    """
    combined_summary = "\n\n".join(summaries)
    messages = [
        {"role": "system", "content": "你是一个帮助用户综合总结多个摘要的助手。请用简洁的中文总结以下内容。"},
        {"role": "user", "content": f"请阅读以下摘要并提供一个综合性的总结：\n\n{combined_summary}"}
    ]
    final_summary = call_openai_chat_completion(messages)
    if final_summary:
        logging.info(f"智能体3：综合总结结果：{final_summary}")
    else:
        logging.warning("智能体3：无法生成综合总结。")
    return final_summary if final_summary else "抱歉，我无法生成综合总结。"


def generate_answer(summary, user_query):
    """
    智能体3：使用OpenAI的ChatCompletion模型基于总结的信息生成最终回答。
    回答将分为三个步骤：爬取、总结和回答。
    """
    messages = [
        {"role": "system", "content": "你是一个知识丰富的助手，能够根据总结的信息用简洁的中文回答用户的问题。"},
        {"role": "user", "content": (
            "请按照以下三个步骤回答用户的问题：\n"
            "1. 爬取相关信息。\n"
            "2. 总结信息。\n"
            "3. 基于总结的信息提供最终回答。\n\n"
            f"总结信息：\n{summary}\n\n"
            f"用户问题：{user_query}"
        )}
    ]
    answer = call_openai_chat_completion(messages)
    if answer:
        logging.info(f"智能体3：最终回答：{answer}")
    else:
        logging.warning("智能体3：无法生成最终回答。")
    return answer if answer else "抱歉，我无法生成回答。"


# Streamlit 应用
def main():
    st.title("OnlingGPT")
    st.write("请输入您的问题，系统将自动爬取相关信息并生成回答。")

    user_query = st.text_input("你的问题：", "")

    if st.button("获取回答"):
        if not user_query.strip():
            st.warning("请输入一个有效的问题。")
            return

        with st.spinner("系统正在处理您的请求，请稍等..."):
            try:
                # 智能体1：爬取网页链接
                logging.info("智能体1：正在爬取相关网页...")
                urls = crawl_web(user_query, num_results=5)
                if not urls:
                    st.error("未找到相关网页。")
                    return
                st.success(f"爬取到 {len(urls)} 个相关网页。")

                # 智能体2：获取网页内容
                logging.info("智能体2：正在获取网页内容...")
                contents = []
                for url in urls:
                    content = fetch_page_content(url)
                    if content:
                        contents.append(content)

                if not contents:
                    st.error("未获取到任何网页内容。")
                    return
                st.success(f"成功获取了 {len(contents)} 个网页的内容。")

                # 智能体2：摘要每个网页内容
                logging.info("智能体2：正在摘要网页内容...")
                individual_summaries = []
                for idx, content in enumerate(contents, 1):
                    logging.info(f"智能体2：摘要第 {idx} 个网页的内容...")
                    summary = summarize_individual_content(content)
                    if summary:
                        individual_summaries.append(summary)
                        st.info(f"摘要第 {idx} 个网页的内容完成。")
                    else:
                        individual_summaries.append("抱歉，无法摘要此内容。")
                        st.warning(f"摘要第 {idx} 个网页的内容失败。")

                # 智能体3：综合总结
                logging.info("智能体3：正在生成综合总结...")
                final_summary = summarize_all_contents(individual_summaries)
                logging.info("智能体3：综合总结完成。")
                st.write("### 综合总结：")
                st.write(final_summary)

                # 智能体3：生成最终回答
                logging.info("智能体3：正在生成回答...")
                answer = generate_answer(final_summary, user_query)
                st.write("### 回答：")
                st.write(answer)

            except Exception as e:
                logging.error(f"发生错误：{e}")
                st.error(f"发生错误：{e}")


if __name__ == "__main__":
    main()
