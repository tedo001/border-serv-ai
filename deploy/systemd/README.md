# Bare-metal installation (systemd)

For a BOP node without Docker. Steps assume Debian/Ubuntu.

```bash
# 1. System dependencies (ffmpeg supplies the RTSP/H.264 decoders)
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv ffmpeg libgl1 libglib2.0-0

# 2. Service account — unprivileged; this process has no reason to be root
sudo useradd --system --home /var/lib/ibvap --create-home ibvap
sudo mkdir -p /opt/ibvap /etc/ibvap /var/lib/ibvap/{data,models} /var/log/ibvap
sudo chown -R ibvap:ibvap /var/lib/ibvap /var/log/ibvap

# 3. Install
sudo python3.11 -m venv /opt/ibvap/venv
sudo /opt/ibvap/venv/bin/pip install /path/to/ibvap-1.0.0-py3-none-any.whl

# 4. Configure
sudo cp configs/site.example.yaml /etc/ibvap/site.yaml
sudo nano /etc/ibvap/site.yaml            # cameras, zones, rules

# 5. Secrets — NOT in site.yaml, which gets copied between posts
sudo /opt/ibvap/venv/bin/ibvap secret | sudo tee /etc/ibvap/ibvap.env
sudo chmod 600 /etc/ibvap/ibvap.env
sudo chown ibvap:ibvap /etc/ibvap/ibvap.env

# 6. Validate before starting — catches bad zones and rule references
sudo -u ibvap /opt/ibvap/venv/bin/ibvap validate /etc/ibvap/site.yaml

# 7. Install and start
sudo cp deploy/systemd/ibvap.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ibvap
sudo systemctl status ibvap
```

The bootstrap administrator password is printed **once**, to the journal, on
first start:

```bash
sudo journalctl -u ibvap | grep -A5 "bootstrap administrator"
```

Change it at first login.

## Sizing

Run the benchmark on the actual node hardware before committing cameras to it:

```bash
sudo -u ibvap /opt/ibvap/venv/bin/ibvap benchmark --resolution 1920x1080
```

The report gives a camera count at 8 fps with 30% headroom, sized on p95
latency. Exceeding it does not break the node — frames are dropped rather than
queued, so alert latency stays flat — but coverage falls.

## Log rotation

The platform rotates its own log file (32 MB x 5). If you prefer journald only,
remove `telemetry.log_file` from `site.yaml`.
