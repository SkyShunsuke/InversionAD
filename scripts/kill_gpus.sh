# take input PID as argument and then kill the process by the kill -9 $(seq x x+7)
# usage: kill_gpus.sh <PID>

#!/bin/bash
if [ $# -ne 1 ]; then
    echo "Usage: $0 <PID>"
    exit 1
fi
PID=$1
echo "Killing process with PID: $PID"
kill -9 $PID
for i in $(seq 0 7); do
    kill -9 $(($PID + $i))
done
echo "Killed processes with PIDs: $(seq $PID $(($PID + 7)))"
echo "Done"

