#!/bin/bash
# Quick script to restart the LLM API service

echo "🔄 Restarting LLM API service..."
sudo systemctl restart llm-api.service

echo "⏳ Waiting for service to start..."
sleep 3

echo ""
echo "📊 Service Status:"
systemctl status llm-api.service --no-pager -l | head -20

echo ""
echo "🧪 Testing endpoint..."
curl -s http://127.0.0.1:8080/health | python3 -m json.tool 2>/dev/null || curl -s http://127.0.0.1:8080/health

echo ""
echo "✅ Done!"
