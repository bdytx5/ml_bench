#!/bin/bash
##### NOT WORKING .... yet 
# Check if the speed argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <speed>"
  echo "Example: $0 1.0"
  exit 1
fi

# Check if the interface argument is provided
if [ -z "$2" ]; then
  echo "Usage: $0 <speed> <interface>"
  echo "Example: $0 1.0 en1"
  exit 1
fi

SPEED=$1
INTERFACE=$2

# Convert the speed to the correct format
SPEED_FORMAT="${SPEED}Mbit/s"

# Create and load the dnctl rules
sudo dnctl pipe 1 config bw $SPEED_FORMAT
echo "dummynet in on $INTERFACE from any to any pipe 1" | sudo pfctl -f -
echo "dummynet out on $INTERFACE from any to any pipe 1" | sudo pfctl -f -

# Enable pf
sudo pfctl -E

echo "WiFi speed capped at $SPEED_FORMAT on interface $INTERFACE"


