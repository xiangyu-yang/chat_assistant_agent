import httpx
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class WebSearch:
    def __init__(self):
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        results = []
        try:
            url = "https://www.baidu.com/s"
            params = {
                "wd": query,
                "pn": 0,
                "rn": num_results,
                "ie": "utf-8"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
            }
            
            response = self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            search_results = soup.find_all('div', class_='result')
            
            if not search_results:
                search_results = soup.find_all('div', class_='c-container')
            
            if not search_results:
                search_results = soup.find_all('div', attrs={'tpl': True})
            
            for idx, result in enumerate(search_results[:num_results]):
                try:
                    title_tag = result.find('h3')
                    if not title_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    if not title or len(title) < 2:
                        continue
                    
                    link_tag = title_tag.find('a')
                    if not link_tag or 'href' not in link_tag.attrs:
                        continue
                    
                    url = link_tag['href']
                    if not url.startswith('http'):
                        continue
                    
                    if 'baidu.com/link' in url:
                        try:
                            redirect_response = self.client.get(url, headers=headers, timeout=10.0, follow_redirects=True)
                            url = str(redirect_response.url)
                        except:
                            pass
                    
                    summary_tag = result.find('div', class_='c-abstract')
                    if not summary_tag:
                        summary_tag = result.find('p', class_='op_exactqa_s_answer')
                    if not summary_tag:
                        summary_tag = result.find('span', class_='content-right_8Zs40')
                    if not summary_tag:
                        summary_tag = result.find('div', class_='result-op')
                    if not summary_tag:
                        summary_tag = result.find('span', class_='newTimeFactor_before_abs m')
                    if not summary_tag:
                        summary_tag = result.find('p')
                    
                    summary = summary_tag.get_text(strip=True) if summary_tag else ""
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "rank": idx + 1
                    })
                except Exception as e:
                    logger.warning(f"解析搜索结果失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
        
        return results
    
    def fetch_page_content(self, url: str, max_length: int = 1000) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
            }
            
            response = self.client.get(url, headers=headers, timeout=15.0)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            text_content = soup.get_text(separator=' ', strip=True)
            
            return text_content[:max_length] if len(text_content) > max_length else text_content
            
        except Exception as e:
            logger.error(f"获取网页内容失败 {url}: {e}")
            return ""
    
    def search_with_content(self, query: str, num_results: int = 3) -> List[Dict]:
        results = self.search(query, num_results)
        
        for result in results:
            content = self.fetch_page_content(result['url'])
            result['content'] = content
        
        return results
    
    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False