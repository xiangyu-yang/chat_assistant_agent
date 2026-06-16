import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from src.config.settings import SESSIONS_DIR

logger = logging.getLogger(__name__)


def load_sessions() -> List[Dict]:
    sessions = []
    if os.path.exists(SESSIONS_DIR):
        for filename in os.listdir(SESSIONS_DIR):
            if filename.endswith('.json'):
                session_id = filename[:-5]
                filepath = os.path.join(SESSIONS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sessions.append({
                            'id': session_id,
                            'title': data.get('title', '未命名会话'),
                            'created_at': data.get('created_at', ''),
                            'updated_at': data.get('updated_at', '')
                        })
                except Exception as e:
                    logger.error(f"Error loading session {session_id}: {e}")
    sessions.sort(key=lambda x: x['updated_at'], reverse=True)
    return sessions


def load_session(session_id: str) -> Optional[Dict]:
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
    return None


def save_session(session_id: str, chat_history: List[Dict], title: Optional[str] = None):
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    session_data = {
        'title': title or chat_history[0]['content'][:16] + '...' if chat_history else '未命名会话',
        'chat_history': chat_history,
        'created_at': datetime.now().isoformat() if not os.path.exists(filepath) else load_session(session_id).get('created_at', datetime.now().isoformat()),
        'updated_at': datetime.now().isoformat()
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")


def delete_session(session_id: str):
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)


def generate_session_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')