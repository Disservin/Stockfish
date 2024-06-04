#!/bin/bash

./stockfish <<EOF
setoption name Threads value 30
setoption name Hash value 26000
quit
EOF
