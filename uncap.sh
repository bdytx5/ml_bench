#!/bin/bash

# Disable pf
sudo pfctl -d

# Clear pf rules
sudo pfctl -F all

# Clear dnctl rules
sudo dnctl -q flush

echo "WiFi speed cap removed"

