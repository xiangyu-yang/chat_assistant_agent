import requests
import base64
import subprocess
import time
import os
import platform

KKFILEVIEW_URL = "http://localhost:8012"

def is_kkfileview_available():
    try:
        response = requests.get(f"{KKFILEVIEW_URL}/index", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_docker_desktop():
    """
    启动 Docker Desktop（macOS）
    返回: (success: bool, message: str)
    """
    # 仅支持macOS
    if platform.system() != 'Darwin':
        return False, "❌ 当前仅支持 macOS 系统自动启动 Docker Desktop"
    
    try:
        # 方法1: 使用open命令（推荐）
        result = subprocess.run(
            ['open', '-a', 'Docker'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return True, "✅ Docker Desktop 启动命令已发送！\n\n💡 请等待Docker Desktop完全启动（可能需要1-2分钟），然后再次尝试启动KKFileView服务。"
        else:
            # 方法2: 直接执行Docker应用
            docker_app_path = '/Applications/Docker.app/Contents/MacOS/Docker'
            if os.path.exists(docker_app_path):
                subprocess.Popen(
                    [docker_app_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True, "✅ Docker Desktop 启动命令已发送！\n\n💡 请等待Docker Desktop完全启动，然后再次尝试。"
            else:
                return False, f"❌ 无法找到 Docker 应用。\n\n📍 请手动启动：在 Finder 中打开 /Applications/Docker.app"
                
    except FileNotFoundError:
        return False, "❌ 找不到 Docker 应用。\n\n📥 请前往 https://www.docker.com/products/docker-desktop/ 下载安装"
    except subprocess.TimeoutExpired:
        return False, "❌ 启动命令超时，请手动启动 Docker Desktop"
    except Exception as e:
        return False, f"❌ 启动 Docker Desktop 失败: {str(e)}"

def check_docker_status():
    """
    检查Docker状态
    返回: (is_installed: bool, is_running: bool, message: str)
    """
    try:
        # 检查Docker是否安装
        result = subprocess.run(['docker', '--version'], 
                             capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, False, "Docker未安装。请前往 https://www.docker.com 下载并安装 Docker Desktop。"
        
        # 检查Docker守护进程是否运行
        result = subprocess.run(['docker', 'info'], 
                             capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            if "Cannot connect to the Docker daemon" in result.stderr or \
               "Is the docker daemon running?" in result.stderr:
                return True, False, "⚠️ Docker已安装但未运行。"
            return True, False, f"Docker状态异常: {result.stderr}"
        
        return True, True, "Docker运行正常"
        
    except FileNotFoundError:
        return False, False, "Docker未安装。请前往 https://www.docker.com 下载并安装 Docker Desktop。"
    except subprocess.TimeoutExpired:
        return False, False, "Docker检查超时，请确保Docker Desktop已启动。"
    except Exception as e:
        return False, False, f"检查Docker状态时出错: {str(e)}"

def start_kkfileview_service(progress_callback=None):
    """
    启动KKFileView服务
    
    参数:
        progress_callback: 进度回调函数，接收字符串参数用于显示进度信息
    
    返回: (success: bool, message: str)
    """
    if is_kkfileview_available():
        return True, "✅ KKFileView 服务已经在运行中"
    
    # 检查Docker状态
    is_installed, is_running, docker_message = check_docker_status()
    
    if not is_installed:
        return False, f"❌ {docker_message}\n\n📥 下载地址: https://www.docker.com/products/docker-desktop/"
    
    if not is_running:
        return False, f"❌ {docker_message}\n\n💡 提示: 在应用程序中找到 Docker Desktop 并启动它，然后再次尝试。"
    
    # 检查是否已有容器在运行
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=kkfileview', '--format', '{{.Names}}'],
            capture_output=True, text=True, timeout=5
        )
        container_exists = 'kkfileview' in result.stdout
        
        if container_exists:
            # 如果容器已存在但未运行，则启动它
            if progress_callback:
                progress_callback("正在启动现有容器...")
            subprocess.run(['docker', 'start', 'kkfileview'], 
                         capture_output=True, timeout=30)
            time.sleep(2)  # 等待服务启动
        else:
            # 创建并启动新容器
            if progress_callback:
                progress_callback("🛠️ 首次启动，正在创建KKFileView容器（需要下载镜像，请耐心等待）...")
            docker_command = [
                'docker', 'run', '-d',
                '--name', 'kkfileview',
                '-p', '8012:8012',
                '-v', '/Users/george/Documents/开发/develop/mini_assistant/knowledge_bases:/opt/kkfileview/knowledge_bases',
                '--restart=always',
                'keenq/kkfileview:4.1.0'
            ]
            result = subprocess.run(docker_command, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return False, f"❌ 创建容器失败: {result.stderr}"
            
            # 等待服务启动
            time.sleep(5)
        
        # 验证服务是否启动成功
        for _ in range(10):  # 最多等待30秒
            if is_kkfileview_available():
                return True, "✅ KKFileView 服务启动成功！"
            time.sleep(3)
        
        return False, "⚠️ KKFileView 服务启动超时。请在终端运行以下命令查看日志：\n```bash\ndocker logs kkfileview\n```"
        
    except subprocess.TimeoutExpired:
        return False, "❌ 启动命令执行超时，请检查网络连接后重试。"
    except Exception as e:
        return False, f"❌ 启动失败: {str(e)}"

def stop_kkfileview_service():
    """
    停止KKFileView服务
    返回: (success: bool, message: str)
    """
    try:
        # 停止容器
        subprocess.run(['docker', 'stop', 'kkfileview'], 
                      capture_output=True, timeout=10)
        
        # 验证是否已停止
        if not is_kkfileview_available():
            return True, "KKFileView 服务已停止"
        else:
            return False, "服务停止失败"
            
    except Exception as e:
        return False, f"停止服务失败: {str(e)}"

def get_file_preview_url(file_path):
    container_path = file_path.replace(
        "/Users/george/Documents/开发/develop/mini_assistant/knowledge_bases",
        "/opt/kkfileview/knowledge_bases"
    )
    file_url = f"file://{container_path}"
    encoded_url = base64.b64encode(file_url.encode('utf-8')).decode('utf-8')
    return f"{KKFILEVIEW_URL}/onlinePreview?url={encoded_url}"
