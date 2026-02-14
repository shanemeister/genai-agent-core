#!/bin/bash
# Firewall rules to restrict port 8080 access
# Only allow: localhost, 192.168.4.22, 192.168.4.25, and Cloudflare tunnel

echo "Setting up firewall rules for port 8080..."

# First, remove any existing rules for port 8080
sudo iptables -D INPUT -p tcp --dport 8080 -j ACCEPT 2>/dev/null
sudo iptables -D INPUT -p tcp --dport 8080 -j DROP 2>/dev/null

# Allow localhost (127.0.0.1)
sudo iptables -I INPUT -p tcp -s 127.0.0.1 --dport 8080 -j ACCEPT

# Allow your Mac (192.168.4.22)
sudo iptables -I INPUT -p tcp -s 192.168.4.22 --dport 8080 -j ACCEPT

# Allow the server itself (192.168.4.25)
sudo iptables -I INPUT -p tcp -s 192.168.4.25 --dport 8080 -j ACCEPT

# Allow localhost IPv6
sudo ip6tables -I INPUT -p tcp -s ::1 --dport 8080 -j ACCEPT 2>/dev/null

# Drop all other connections to port 8080
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP

echo "✅ Firewall rules applied!"
echo ""
echo "Allowed IPs for port 8080:"
echo "  - 127.0.0.1 (localhost)"
echo "  - 192.168.4.22 (your Mac)"
echo "  - 192.168.4.25 (server itself)"
echo ""
echo "Current rules for port 8080:"
sudo iptables -L INPUT -n | grep -A 5 "dpt:8080"
