#!/bin/bash

cd "$(dirname "$0")"

echo "正在启动智能对话助理服务..."
echo "端口: 8501"
echo "访问地址: http://localhost:8501"
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0