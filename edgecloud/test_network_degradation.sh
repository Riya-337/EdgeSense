#!/bin/bash

# Ensure script is run with sudo for tc commands
if [ "$EUID" -ne 0 ]; then
  echo "[!] Please run with sudo: sudo ./test_network_degradation.sh"
  exit 1
fi

INTERFACE="lo"  # Loopback interface for local testing (use eth0/wlan0 for WAN)

show_menu() {
    echo "========================================================="
    echo "     EDGESENSE NETWORK DEGRADATION SIMULATOR            "
    echo "========================================================="
    echo " 1) Apply 150ms Artificial Delay to Local Network"
    echo " 2) Apply 30% Packet Loss to Local Network"
    echo " 3) Apply Heavy Jitter (100ms ± 50ms delay)"
    echo " 4) Clear ALL Network Degradation Rules"
    echo " 5) Check Current Traffic Control (tc) Rules"
    echo " 6) Exit"
    echo "========================================================="
}

while true; do
    show_menu
    read -p "Select an option [1-6]: " choice
    case $choice in
        1)
            echo "--> Injecting 150ms delay on $INTERFACE..."
            tc qdisc del dev $INTERFACE root 2>/dev/null
            tc qdisc add dev $INTERFACE root netem delay 150ms
            echo "[✓] Applied 150ms latency spike."
            ;;
        2)
            echo "--> Injecting 30% packet loss on $INTERFACE..."
            tc qdisc del dev $INTERFACE root 2>/dev/null
            tc qdisc add dev $INTERFACE root netem loss 30%
            echo "[✓] Applied 30% packet loss."
            ;;
        3)
            echo "--> Injecting variable jitter (100ms ± 50ms) on $INTERFACE..."
            tc qdisc del dev $INTERFACE root 2>/dev/null
            tc qdisc add dev $INTERFACE root netem delay 100ms 50ms
            echo "[✓] Applied jitter."
            ;;
        4)
            echo "--> Clearing network degradation rules..."
            tc qdisc del dev $INTERFACE root 2>/dev/null
            echo "[✓] Network restored to normal."
            ;;
        5)
            echo "--> Active Rules on $INTERFACE:"
            tc qdisc show dev $INTERFACE
            ;;
        6)
            echo "Exiting network simulator."
            exit 0
            ;;
        *)
            echo "Invalid option. Please choose 1-6."
            ;;
    esac
    echo ""
done
