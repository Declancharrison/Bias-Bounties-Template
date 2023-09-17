#!/usr/bin/expect -f

set timeout -1
spawn ssh-keygen -t ed25519
expect {
    ":" {
        send "\r"  ;# Send an empty input (Enter key)
        exp_continue ;# Continue to expect the next ":" occurrence
    }
    eof {
        exit
    }
}