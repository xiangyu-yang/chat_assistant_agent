import httpx
from typing import Optional, List, Dict
import logging
import json

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3.6:35b-a3b-q8_0",
        api_key: str = "ollama",
        max_history_messages: int = 20,
        max_history_tokens: int = 8192
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(timeout=300.0)
        self.history: List[Dict] = []
        self.max_history_messages = max_history_messages
        self.max_history_tokens = max_history_tokens
    
    def _count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def _trim_history(self):
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages:]
            logger.info(f"Trimmed history to {self.max_history_messages} messages")
        
        total_tokens = sum(self._count_tokens(msg.get("content", "")) for msg in self.history)
        while total_tokens > self.max_history_tokens and len(self.history) > 0:
            removed_msg = self.history.pop(0)
            total_tokens -= self._count_tokens(removed_msg.get("content", ""))
            logger.info(f"Removed message due to token limit, remaining tokens: {total_tokens}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling LLM API: {e}")
            return f"Error: Unable to connect to LLM service. Please check if Ollama is running."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        context_text = "\n\n".join([f"【文档 {i+1}】：\n{doc}" for i, doc in enumerate(context)])

        if system_prompt is None:
            system_prompt = """你是一名实用的人工智能助手。你将收到一个问题以及来自文档的相关上下文信息。

你的任务：
1. 仅依据所提供的上下文回答问题
2. 若上下文信息不足以完整回答问题，需清晰说明现有可用信息
3. 回答需精准，尽可能标注支撑答案的文档来源
4. 若在上下文中未找到相关信息，直接说明即可

重要要求：不得编造上下文中不存在的信息。

使用中文输出思考过程。
"""

        prompt = f"""上下文信息：
{context_text}

问题：{query}

请根据以上上下文提供有帮助的回答。"""

        return self.generate(prompt, system_prompt=system_prompt)

    def chat(
        self,
        query: str,
        system_prompt: Optional[str] = None
    ) -> str:
        if system_prompt is None:
            system_prompt = """你是一个帮助用户解答问题的AI助手。

思考过程格式要求：
1. 在回答之前，先输出你的思考过程
2. 思考过程用【深度思考】标签包裹
3. 思考过程要详细，包括：分析问题、搜索相关知识、整理思路等
4. 思考完成后再给出最终回答

输出格式示例：
【深度思考】
用户问的是...
我需要分析...
我已经获得了...
现在可以组织回答...
【/深度思考】

最终回答：
(你的详细回答内容)
"""

        return self.generate(query, system_prompt=system_prompt)
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        think: bool = True,
        use_history: bool = True,
        feedback_history: Optional[List[Dict]] = None
    ):
        messages = []

        feedback_prompt = ""
        if feedback_history and len(feedback_history) > 0:
            feedback_items = []
            for i, feedback in enumerate(feedback_history[-5:]):
                user_query = feedback.get('query', '')
                assistant_response = feedback.get('response', '')
                rating = feedback.get('rating', 0)
                if rating > 0:
                    feedback_items.append(f"【好评示例{i+1}】\n用户问：{user_query}\n你的回答：{assistant_response}\n用户评价：有用（点赞）")
                elif rating < 0:
                    feedback_items.append(f"【改进示例{i+1}】\n用户问：{user_query}\n你的回答：{assistant_response}\n用户评价：无用（点踩），请改进回答方式")
            
            if feedback_items:
                feedback_prompt = "\n\n以下是用户对之前回答的反馈，请注意学习：\n" + "\n\n".join(feedback_items)
                logger.info(f"Added feedback prompt with {len(feedback_items)} feedback items")

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt + feedback_prompt
            })

        if use_history and self.history:
            logger.info(f"Adding {len(self.history)} history messages")
            messages.extend(self.history)

        messages.append({
            "role": "user",
            "content": prompt
        })

        logger.info(f"Total messages in request: {len(messages)}")
        base_url_no_v1 = self.base_url.replace("/v1", "")
        api_url = f"{base_url_no_v1}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }

        logger.info(f"Starting stream request to {api_url}")
        
        try:
            with httpx.stream(
                "POST",
                api_url,
                json=payload,
                timeout=120.0,
                follow_redirects=True
            ) as response:
                response.raise_for_status()
                logger.info("Connected to streaming API, waiting for chunks...")
                
                full_content = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message") or {}
                        content = msg.get("content", "")
                        full_content += content
                        
                        yield {
                            "content": content,
                            "thinking": msg.get("thinking", ""),
                            "done": chunk.get("done", False)
                        }
                        
                        if chunk.get("done"):
                            self.history.append({"role": "user", "content": prompt})
                            self.history.append({"role": "assistant", "content": full_content})
                            self._trim_history()
                            logger.info(f"Received done, stream completed. History has {len(self.history)} messages")
                            return
                            
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error: {e}")
                        continue
                    
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling LLM API: {e}")
            yield {"content": f"Error: Unable to connect to LLM service. Please check if Ollama is running.", "thinking": "", "done": True}
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield {"content": f"Error: {str(e)}", "thinking": "", "done": True}
    
    def chat_stream(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        think: bool = True,
        use_history: bool = True,
        feedback_history: Optional[List[Dict]] = None
    ):
        if system_prompt is None:
            system_prompt = """你是一个有深度思考能力的AI助手。请根据用户的问题给出准确、详细的回答。深度思考过程必须全部使用中文，禁止使用任何英文。"""

        return self.generate_stream(query, system_prompt=system_prompt, think=think, use_history=use_history, feedback_history=feedback_history)
    
    def generate_with_context_stream(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        think: bool = True,
        use_history: bool = True,
        feedback_history: Optional[List[Dict]] = None
    ):
        context_text = "\n\n".join([f"【文档 {i+1}】：\n{doc}" for i, doc in enumerate(context)])

        if system_prompt is None:
            system_prompt = f"""你是一个有深度思考能力的AI助手。请根据提供的上下文信息，回答用户的问题。如果上下文不足以回答问题，请明确说明。深度思考过程必须全部使用中文，禁止使用任何英文。

上下文信息：
{context_text}"""

        return self.generate_stream(query, system_prompt=system_prompt, think=think, use_history=use_history, feedback_history=feedback_history)

    def clear_history(self):
        self.history = []
    
    def test_connection(self, model_name: Optional[str] = None) -> Dict:
        target_model = model_name if model_name else self.model
        
        try:
            base_url_no_v1 = self.base_url.replace("/v1", "")
            
            model_list_response = self.client.get(
                f"{base_url_no_v1}/api/tags",
                timeout=5.0
            )
            
            if model_list_response.status_code != 200:
                return {
                    "success": False,
                    "message": f"❌ 无法获取模型列表 (状态码: {model_list_response.status_code})",
                    "status": "service_error"
                }
            
            model_data = model_list_response.json()
            available_models = [m.get("name", "") for m in model_data.get("models", [])]
            
            if target_model not in available_models:
                return {
                    "success": False,
                    "message": f"❌ 模型 '{target_model}' 未找到。可用模型: {', '.join(available_models) if available_models else '无'}",
                    "status": "model_not_found"
                }
            
            test_payload = {
                "model": target_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            test_response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=test_payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            if test_response.status_code == 200:
                result = test_response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return {
                        "success": True,
                        "message": f"✅ 连接成功！模型 '{target_model}' 响应正常",
                        "status": "connected"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"❌ 模型 '{target_model}' 响应格式异常",
                        "status": "response_format_error"
                    }
            else:
                try:
                    error_data = test_response.json()
                    error_msg = error_data.get("error", {}).get("message", f"状态码: {test_response.status_code}")
                except:
                    error_msg = f"状态码: {test_response.status_code}"
                
                return {
                    "success": False,
                    "message": f"❌ 模型 '{target_model}' 响应异常: {error_msg}",
                    "status": "model_error"
                }
                    
        except httpx.ConnectError:
            return {
                "success": False,
                "message": "❌ 无法连接到服务，请确认服务地址正确且Ollama正在运行",
                "status": "connection_refused"
            }
        except httpx.TimeoutException:
            return {
                "success": False,
                "message": "❌ 连接超时，模型可能正在加载或服务异常",
                "status": "timeout"
            }
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return {
                "success": False,
                "message": f"❌ 连接测试失败: {str(e)}",
                "status": "error"
            }

    def get_available_models(self) -> List[str]:
        try:
            base_url_no_v1 = self.base_url.replace("/v1", "")
            response = self.client.get(
                f"{base_url_no_v1}/api/tags",
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                models = []
                if "models" in data:
                    for model in data["models"]:
                        models.append(model.get("name", ""))
                return models
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Failed to get model list: {e}")
            return []

    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def create_llm_client(
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen3.6:35b-a3b-q8_0",
    api_key: str = "ollama"
) -> LLMClient:
    return LLMClient(base_url=base_url, model=model, api_key=api_key)