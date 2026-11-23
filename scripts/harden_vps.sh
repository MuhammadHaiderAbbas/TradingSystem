#!/usr/bin/env bash
# Baseline hardening for a private Ubuntu/Debian VPS running Sentinel.
# READ IT BEFORE RUNNING. It disables SSH password login: make sure your SSH key works first.
#   sudo ./scripts/harden_vps.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root"; exit 1; }

SSH_PORT="${SSH_PORT:-22}"

apt-get update
apt-get install -y ufw fail2ban unattended-upgrades chrony

# Firewall: deny everything inbound except SSH. The dashboard is NOT exposed;
# use an SSH tunnel (ssh -L 8765:127.0.0.1:8765 user@host) or a VPN (WireGuard/Tailscale).
ufw default deny incoming
ufw default allow outgoing
ufw allow "${SSH_PORT}/tcp"
ufw --force enable

# SSH: keys only, no root login.
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?KbdInteractiveAuthentication.*/KbdInteractiveAuthentication no/' /etc/ssh/sshd_config
systemctl reload ssh || systemctl reload sshd

# fail2ban for sshd
cat > /etc/fail2ban/jail.d/sshd.local <<EOF
[sshd]
enabled = true
port = ${SSH_PORT}
maxretry = 5
bantime = 1h
EOF
systemctl enable --now fail2ban

# Time sync (exchanges reject requests with skewed timestamps)
systemctl enable --now chrony

# Automatic security updates
dpkg-reconfigure -f noninteractive unattended-upgrades

# Dedicated unprivileged user
id sentinel >/dev/null 2>&1 || useradd --system --create-home --home-dir /opt/sentinel --shell /usr/sbin/nologin sentinel

echo "Done. Next: deploy Sentinel to /opt/sentinel as user 'sentinel' and install scripts/sentinel.service."
